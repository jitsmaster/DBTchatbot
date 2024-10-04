export class WebSocketRequest {

	constructor(public text: string,
		public taxonomy: any,
		public temperature: number = 0,
		public leafOnly: boolean = false
	) { }
}

export class WebSocketEmbeddingUpsertRequest {
	constructor(
		public upsertContent: {
			content: string;
			metadata: Record<string, string | number | boolean>;
		}[],
		public collectionName: string, public source: string,
		public format: string,
		public selector: string = "") { }
}

export class WebSocketEmbeddingRAGSearchRequest {
	constructor(public searchKeywords: string, public collectionName: string = "",
		public source: string = "",
		public topKCount: number | undefined = 10,
		public fragSize: number | undefined = 60) { }
}

export class WebSocketEmbeddingRAGQueryRequest {

	constructor(public query: string, public collectionName: string = "",
		public source: string = "",
		public topKCount: number | undefined = 10) { }
}

export class WebSocketEmbeddingRAGQnARequest {
	constructor(public question: string,
		public minSimilarity: number,
		public k: number,
		public collectionName: string = "",
		public source: string = "",
		public additionalInstructions: string = "") { }
}

export class DocumentExtractionFilterBase {
	constructor(
		public format: string,
		public metadata: {
			source: string
		},
	) { }
}
export class HtmlFilter extends DocumentExtractionFilterBase {
	constructor(
		public cssSelector: string,
		override format: string,
		override metadata: any,
	) {
		super(format, metadata);
	}
}

export class XmlFilter extends DocumentExtractionFilterBase {
	constructor(
		public xpath: string,
		override format: string,
		override metadata: any,
	) {
		super(format, metadata);
	}
}