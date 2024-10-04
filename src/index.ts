import express, { Request, Response } from 'express';
import { ChromaVectorStore } from './ChromaVectorStore'

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json());

const langchain = new Langchain();
const chromaDB = new ChromaDB();

// Endpoint to create vectors and upsert into ChromaDB
app.post('/upsert', async (req: Request, res: Response) => {
  try {
    const { text } = req.body;
    const vector = await langchain.createVector(text);
    await chromaDB.upsert(vector);
    res.status(200).send('Vector upserted successfully');
  } catch (error) {
    res.status(500).send('Error upserting vector');
  }
});

// Endpoint to perform QnA based on ChromaDB query result
app.post('/qna', async (req: Request, res: Response) => {
  try {
    const { question } = req.body;
    const results = await chromaDB.query(question);
    const answer = await langchain.qna(question, results);
    res.status(200).json({ answer });
  } catch (error) {
    res.status(500).send('Error performing QnA');
  }
});

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});