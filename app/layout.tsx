import "./globals.css";
import classNames from "classnames";

import { Geologica } from "next/font/google";
import { APP_TITLE } from "./constants";

const ubuntu = Geologica({
  subsets: ["latin"],
  weight: ["400"],
});

export const metadata = {
  title: APP_TITLE,
  description: "Local GPT",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body
        className={classNames(
          ubuntu.className,
          "bg-stone-500",
          "text-white",
          "text-4xl"
        )}
      >
        {children}
      </body>
    </html>
  );
}
