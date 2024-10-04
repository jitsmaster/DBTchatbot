import fs from 'fs';
import pdf from 'pdf-parse';

async function loadPdf(filePath: string): Promise<string> {
    const dataBuffer = fs.readFileSync(filePath);
    try {
        const data = await pdf(dataBuffer);
        return data.text;
    } catch (error) {
        console.error("Failed to load PDF", error);
        throw error;
    }
}

export default loadPdf;