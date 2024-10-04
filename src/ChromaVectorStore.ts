import { Chroma } from "@langchain/community/vectorstores/chroma";
import { Document } from "@langchain/core/documents";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import {
	RunnablePassthrough,
	RunnableSequence,
} from "@langchain/core/runnables";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import * as cheerio from "cheerio";
import { ChromaClient } from "chromadb";
import type { BaseDocumentLoader } from "langchain/document_loaders/base";
import { ScoreThresholdRetriever } from "langchain/retrievers/score_threshold";
import { formatDocumentsAsString } from "langchain/util/document";
import MarkdownIt from "markdown-it";
import xml2js, { parseStringPromise } from "xml2js";
import { DOMParser } from "xmldom";
import xpath from "xpath";
import { iterableArray } from "./IterateUntil";
import { HtmlFilter, XmlFilter, type DocumentExtractionFilterBase } from "./WebSocketModels";
import { OllamaEmbeddings } from "@langchain/ollama";


export class ChromaVectorStore {
	embedding: OllamaEmbeddings;
	splitter: RecursiveCharacterTextSplitter;
	chromaStore: Chroma;
	constructor(
		private gpt: ChatOpenAI,
		private chromaEndpoint: string,
		private collectionName: string,
		private maxQueryConcurrency: number = 4) {
		this.embedding = new OllamaEmbeddings({
			model: "llama3.2"
		});

		this.splitter = new RecursiveCharacterTextSplitter({
			chunkSize: parseInt(process.env.EMBEDDING_SPLITTER_CHUNK_SIZE ?? "8192"), //min size for 4o 
			chunkOverlap: 128,
			keepSeparator: true
		});

		//using cosine distance function, to generate most accurate result for chatbot
		/**
		 * Formula: 
		 * 1.0−(∑(Ai * Bi)/sqrt(∑(Ai^2)) * sqrt(∑(Bi^2))))
		 */
		this.chromaStore = new Chroma(this.embedding, {
			collectionName: this.collectionName,
			url: this.chromaEndpoint,
			collectionMetadata: {
				"hnsw:space": "cosine"
			}
		});
	}

	loaders: BaseDocumentLoader[] = [];

	addLoader(loader: BaseDocumentLoader) {
		this.loaders.push(loader);
	}

	async upsert() {
		//do loading one at a time,
		//since they could end up taking a lot of memory
		this.chromaStore.ensureCollection();
		for (const loader of this.loaders) {
			const docs = await loader.load();
			const chunks = await this.splitter.splitDocuments(docs);
			chunks.forEach(chunk => {
				chunk.metadata = {
					"source": loader.constructor.name
				};
			});

			const res = await this.chromaStore.addDocuments(chunks);
			return res;
		}
	}

	async upsertContent(contents: {
		content: string;
		metadata?: any;
	}[], filter: DocumentExtractionFilterBase) {
		const processedContent = await Promise.all(contents
			.map(async content => {
				if (filter.format === "html") {
					try {
						const $ = cheerio.load(content.content);
						const node = (filter as HtmlFilter).cssSelector ?
							$((filter as HtmlFilter).cssSelector) :
							$("body");
						return {
							content: node.text(),
							metadata: content.metadata
						};
					}
					catch (e) {
						throw new Error("Invalid HTML content, cannot be parsed by Cheerio.");
					}
				}
				else if (filter.format === "md") {
					try {
						const md = new MarkdownIt();
						const doc = cheerio.load(md.render(content.content));
						//no need to filter out anything in md
						return {
							content: doc.text(),
							metadata: content.metadata
						};
					}
					catch (e) {
						throw new Error("Invalid Markdown content, cannot be parsed by MarkdownIt.");
					}
				}
				else if (filter.format === "xml") {
					try {
						const res = await parseStringPromise(content.content);
						const resObj = new xml2js.Builder().buildObject(res);

						//use cleaned up xml
						const doc = new DOMParser().parseFromString(resObj, "text/xml");
						const node = xpath.select((filter as XmlFilter).xpath, doc, true) as Node;

						if (!node) {
							const errMsg = "Error loading xml content: XPath query returned no results.";
							console.log(errMsg);
							throw new Error(errMsg);
						}

						return {
							content: node.textContent,
							metadata: content.metadata
						};
					}
					catch (e) {
						throw new Error("Invalid XML content, cannot be parsed by xml2js.");
					}
				}
				else {
					return content;
				}
			}));

		const source = filter.metadata?.source ?? "";

		await this.chromaStore.ensureCollection();
		const docs = [];

		//merge together all the content
		const chunks = (await Promise.all(processedContent
			.map(async content => {
				const contentResult = await content;
				if (!contentResult || !contentResult.content)
					return [];
				const chunkObjs = (await this.splitter.splitText(contentResult.content))
					.map(chunk => ({
						content: chunk,
						metadata: contentResult.metadata
					}));

				return chunkObjs;
			})))
			.flat();

		docs.push(...chunks
			.map(chunk => new Document({
				pageContent: chunk.content,
				metadata: {
					...chunk.metadata,
					"source": source
				}
			})));

		const res = await this.chromaStore.addDocuments(docs);
		console.log(`Upserted ${docs.length} documents from ${source}, result: ${res}`);
	}

	async clear() {
		//have to do this manually
		const chroma = new ChromaClient({
			path: this.chromaEndpoint
		});

		await chroma.deleteCollection({
			name: this.collectionName
		});
	}

	async query(query: string, source: string = "", k: number = 10) {
		await this.chromaStore.ensureCollection();
		const filter = {} as any;
		if (!!source)
			filter.source = source;

		return await this.chromaStore
			.asRetriever({
				verbose: true,
				filter,
				k: k,
				searchType: "similarity"
			})
			.invoke(query, {
				maxConcurrency: this.maxQueryConcurrency
			});
	}

	async qnA(question: string, minSimilarity: number, k: number, source: string = "", additionalInstructions: string = "") {
		await this.chromaStore.ensureCollection();
		const filter = {
		} as any;
		if (!!source)
			filter.source = source;

		const retriever = !!minSimilarity ?
			ScoreThresholdRetriever.fromVectorStore(
				this.chromaStore,
				{
					minSimilarityScore: minSimilarity,
					filter,
					maxK: k ?? 10,
					searchType: "similarity",
					verbose: true
				}
			) :
			await this.chromaStore
				.asRetriever({
					verbose: true,
					filter,
					k: k ?? 10,
					searchType: "similarity"
				});

		const addiIntrucTemplate = (!!additionalInstructions?.trim()) ?
			`Additional Instructions: ${additionalInstructions}` : "";


		const promptTemplate = `You are an assistant for question-answering tasks. 
			Use the following pieces of retrieved context to answer the question. 
			If you don't know the answer, just say that you don't know. 
			If the context is html, make sure to include the id of the page, replace the part of the url from # character.
			If the context is code, make sure to include sample code blocks.
			Also, make sure to follow the additional instructions on providing the answers, if the additional instructions are provided.
			Return the answer in MarkDown format.
			${addiIntrucTemplate}
			Question: {question} 
			Context: {context} 			
			Answer:`;

		const prompt = ChatPromptTemplate.fromTemplate(promptTemplate);

		const ragChain = RunnableSequence.from([
			{
				context: retriever.pipe(formatDocumentsAsString),
				question: new RunnablePassthrough(),
				additionalInstructions: new RunnablePassthrough()
			},
			prompt,
			this.gpt,
			new StringOutputParser(),
		]);

		return await ragChain.stream(question);
	}
}