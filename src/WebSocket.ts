import WebSocket, { WebSocketServer } from "ws";
import { DocumentExtractionFilterBase, HtmlFilter, WebSocketEmbeddingRAGQnARequest, WebSocketEmbeddingRAGQueryRequest, WebSocketEmbeddingRAGSearchRequest, WebSocketEmbeddingUpsertRequest, WebSocketRequest, XmlFilter } from "./models/WebSocketModels";
import { ChromaVectorStore } from "./ChromaVectorStore";
import { ChatOllama } from "@langchain/ollama";

const ollamaModel = process.env.OLLAMA_MODEL_NAME ?? "llama3.2";

export function createWsServer() {

	const wss = new WebSocketServer({ noServer: true });

	const WS_LOG_PREFIX = "WebSocket:"

	wss.on('connection', function connection(ws, request) {
		const headers = request.headers;
		const qs = request.url?.split('?') ?? [];
		const qsApiKey = qs.length > 1 ? qs[1].split('=')[1] : "";

		ws.on('error', console.error);

		ws.on('open', function open() {
			log('connected', WS_LOG_PREFIX);
			ws.send(`Connected at ${Date.now()}`);
		});

		ws.on('close', function close() {
			log('disconnected', WS_LOG_PREFIX);
		});


		ws.on('message', async function message(data) {
			const apiKey = headers['xai-api-key'] as string || qsApiKey || "";
			const orgName = headers['xai-org'] as string || "anonymous";
			let gptKey = headers["xai-gpt-key"] as string || "";
			if (!gptKey || gptKey === "~") {
				gptKey = process.env.DEFAULT_GPT_KEY || "";
			}

			//check if data is correct JSON format
			let msg: any;
			try {
				msg = JSON.parse(data.toString());
			} catch (error) {
				ws.send(`Bad request: invalid JSON format. Details: ${error}`);
				log(`Invalid JSON format: ${error}`, WS_LOG_PREFIX);
				ws.close();
				return;
			}

			if (!!msg.question) {
				await _qna(WS_LOG_PREFIX, msg, ws);
			}
		});

		ws.send('WebSocket connection established');
		log(`WebSocket connection established`, WS_LOG_PREFIX);
	});

	return wss;
}

async function _qna(WS_LOG_PREFIX: string, msg: any, ws: WebSocket) {
	log("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
	log(`Performing QnA...`, WS_LOG_PREFIX);

	try {
		const queryRequest = msg as WebSocketEmbeddingRAGQnARequest;

        const model = new ChatOllama({
			model: ollamaModel,
			streaming: true
		})

		const store = new ChromaVectorStore(model, process.env.CHROMA_ENDPOINT ?? "",
			queryRequest.collectionName,
			4);

		const result = await store.qnA(queryRequest.question, queryRequest.minSimilarity ?? 0,
			queryRequest.k ?? 10,
			queryRequest.source, queryRequest.additionalInstructions);

		for await (const r of result) {
			ws.send(r);
		}

		log(`QnA Done`, WS_LOG_PREFIX);
	} catch (error) {
		log(`Error performing QnA: ${error}`, WS_LOG_PREFIX);
		ws.send(`Error performing QnA: ${error}`);
	}
	finally {
		ws.send("~~~END~~~");
		ws.close();
	}
}

export function log(mesg: any, prefix = "") {
	//Note: uncomment the line below to see the log messages
	console.log(`${prefix}  ${mesg.toString()}`);
}