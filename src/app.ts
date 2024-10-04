import express, { Request, Response } from 'express';
import { ChromaVectorStore } from './ChromaVectorStore'
import { ChatOllama, Ollama } from '@langchain/ollama';
import path from "path"
import fs from "fs"
import loadPdf from './PdfLoader';

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json());

const chromaUrl = process.env.CHROMA_ENDPOINT ?? "";
const ollamaModel = process.env.OLLAMA_MODEL_NAME ?? "llama3.2";

if (!chromaUrl)
	console.error("Chroma URL not provided.");

// Endpoint to create vectors and upsert into ChromaDB
app.post('/upsert', async (req: Request, res: Response) => {
	try {
		const {
			collectionName,
			pdfFiles
		} = req.body;
		const model = new ChatOllama({
			model: ollamaModel,
			streaming: true
		})
		const store = new ChromaVectorStore(model, chromaUrl, collectionName, 4);

		const content = await Promise.all((pdfFiles as string[])
			.map(async file => {
				const fullPath = path.resolve(__dirname, file);
				const parsedContent = await loadPdf(fullPath);
				return {
					content: parsedContent,
					metadata: {
						url: file
					}
				}
			}))

		await store.upsertContent(content, {
			format: "",
			metadata: {
				source: "DBT Handbook"
			}
		});
		res.status(200).send('Vector upserted successfully');
	} catch (error) {
		res.status(500).send('Error upserting vector');
	}
});

// Endpoint to perform QnA based on ChromaDB query result
app.post('/qna', async (req: Request, res: Response) => {
	try {
		const { question, collection } = req.body as { question: string, collection: string };
		const model = new ChatOllama({
			model: ollamaModel,
			streaming: true
		})
		const chroma = new ChromaVectorStore(model, chromaUrl, collection, 4);
		console.info("performing qna!")
		const result = await chroma.qnA(question, 0, 10, "DBT Handbook", "");

		let msg = "";
		for await (const packet of result) {
			msg += packet;
		}

		res.status(200).json({ answer: msg });
	} catch (error) {
		res.status(500).send('Error performing QnA');
	}
});

app.listen(PORT, () => {
	console.log(`Server is running on port ${PORT}`);
});