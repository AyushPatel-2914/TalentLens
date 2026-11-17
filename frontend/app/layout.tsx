import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "TalentLens",
  description: "Explainable job matching with a recruiter agent",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
