import type { IterableReadableStream } from "@langchain/core/utils/stream";

/**
 * Conditionally iterate over a stream until a condition is met.
 * Fix the problem of program doesn't stop when breaking out of for await loop.
 * This generator function can be for await on the caller side.
 * @param stream 
 * @param condition 
 */
export async function* iterateUtil<TIn, TOut>(stream: IterableReadableStream<TIn>,
	condition: (value: TIn) => boolean,
	transformer: (value: TIn) => TOut): AsyncGenerator<Awaited<TOut>> {

	while (true) {
		let item: any;
		try {
			item = await stream.next();
			const value = item.value as TIn;
			yield transformer(value);
		}
		catch (e) {
			//some times the readable stream host it's owner in the middle of the iteration
			//we just break out of it.
			//This will most likely happen post api calls, when feeding back to ai, which
			//we don't need anyway.
			console.warn(e);
			break;
		}

		if (!!condition && condition(item.value)) {
			break;
		}
	}
}

export async function* iterableArray<TIn, TOut>(array: TIn[],
	originalResult: TOut,
	itemTransformer: (value: TIn, arrayIndex: number) => Promise<TOut>) {
	let i = 0;
	//yield original result
	yield originalResult;
	for (const item of array) {
		yield await itemTransformer(item, i);
		i++
	}
}

// export async function* iterableArrayWithIntermiateStreamResult<TIn, TMiddle, TOut>(array: TIn[],
// 	streamTransformer: (value: TIn, arrayIndex: number) => Promise<AsyncGenerator<TMiddle>>,
// 	itemTransformer: (value: TMiddle, arrayIndex: number) => Promise<TOut>) {

// 	let i = 0;
// 	for (const item of array) {
// 		for await (const packet of (await streamTransformer(item, i))) {
// 			yield await itemTransformer(packet, i);
// 		}
// 		i++
// 	}
// }

