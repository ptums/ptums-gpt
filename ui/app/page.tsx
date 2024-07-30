"use client";

import { useChat } from "ai/react";
import { APP_TITLE, CHAT_API_ENDPOINT } from "./constants";
import classNames from "classnames";

export default function Chat() {
  const { messages, input, handleInputChange, handleSubmit } = useChat({
    api: CHAT_API_ENDPOINT,
  });

  const agent = APP_TITLE.replace(/\s+/g, "");

  return (
    <div className="flex flex-col w-full max-w-4xl py-24 mx-auto stretch">
      <h1 className="text-5xl font-bold text-center">{APP_TITLE}</h1>
      <div
        className={classNames({
          "bg-black bg-opacity-75 shadow p-6 text-lg rounded my-12":
            messages.length > 0,
        })}
      >
        {messages.map((m) => (
          <div key={m.id} className="whitespace-pre-wrap mb-4 text-shadow-sm">
            <span
              className={classNames({
                "text-yellow-300": m.role === "user",
                "text-emerald-300": m.role !== "user",
              })}
            >
              {m.role === "user" ? "User: " : `${agent}: `}
            </span>
            <span className="text-normal">{m.content}</span>
          </div>
        ))}
      </div>

      <form onSubmit={handleSubmit} className="w-full max-w-4xl block">
        <input
          className="fixed bottom-0 w-full max-w-4xl p-4 mb-8 border border-gray-300 rounded shadow-xl rounded text-2xl text-black block"
          value={input}
          placeholder="Say something..."
          onChange={handleInputChange}
        />
      </form>
    </div>
  );
}
