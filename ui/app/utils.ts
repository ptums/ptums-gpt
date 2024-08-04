import { Message } from "ai";

export const buildPrompt = (messages: Message[]): string => {
  return (
    messages
      .map((message) => `${message.role}: ${message.content}`)
      .join("\n\n") + "\n\nassistant:"
  );
};
