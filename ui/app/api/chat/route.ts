import { modelName } from "@/app/constants";
import { streamText } from "ai";
import { experimental_buildOpenAssistantPrompt } from "ai/prompts";
import { ollama } from "ollama-ai-provider";

// Allow streaming responses up to 30 seconds
export const maxDuration = 30;

export async function POST(req: Request) {
  try {
    // Extract the `messages` from the body of the request
    const { messages } = await req.json();

    const result = await streamText({
      model: ollama(modelName),
      prompt: experimental_buildOpenAssistantPrompt(messages),
      maxTokens: 1024,
    });

    // Respond with the stream
    return result.toAIStreamResponse();
  } catch (error) {
    return new Response("Error processing the request", { status: 500 });
  }
}
