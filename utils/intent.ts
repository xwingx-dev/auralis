export type Intent = "transcript-only" | "cloud-text" | "cloud-vision";

export type RoutingDecision = {
  intent: Intent;
  transcriptionOnly: boolean;
  requiresCloud: boolean;
  requiresVision: boolean;
  similarity: number;
  margin: number;
  reason: "matched" | "uncertain" | "empty" | "too-long";
};

export const MAX_TRANSCRIPT_LENGTH = 1200;
export const MIN_SIMILARITY = 0.40;
export const MIN_MARGIN = 0.08;

export const INTENT_EXAMPLES: { intent: Intent; text: string }[] = [
  { intent: "transcript-only", text: "Just transcribe what I say. Do not answer or analyze it." },
  { intent: "transcript-only", text: "Take a note: buy milk and bread on the way home." },
  { intent: "transcript-only", text: "Record this dictation as plain text." },
  { intent: "transcript-only", text: "The meeting starts tomorrow at ten in the morning." },
  { intent: "transcript-only", text: "I went for a walk and had lunch with a friend." },
  { intent: "transcript-only", text: "Write down exactly what I said about this picture, without describing it." },
  { intent: "cloud-text", text: "Explain why the sky is blue." },
  { intent: "cloud-text", text: "What is the capital of France?" },
  { intent: "cloud-text", text: "Summarize these meeting notes and list action items." },
  { intent: "cloud-text", text: "Translate this sentence into Spanish." },
  { intent: "cloud-text", text: "Help me plan a trip and suggest an itinerary." },
  { intent: "cloud-text", text: "Calculate fifteen percent of two hundred." },
  { intent: "cloud-text", text: "Explain how cameras and computer vision work." },
  { intent: "cloud-text", text: "Answer a general knowledge question for me." },
  { intent: "cloud-text", text: "Explain the science behind a natural phenomenon." },
  { intent: "cloud-text", text: "Why is rest important for our health?" },
  { intent: "cloud-text", text: "Help me troubleshoot a technical problem with my laptop." },
  { intent: "cloud-text", text: "Add these numbers and give me the result." },
  { intent: "cloud-text", text: "Compare these ideas and recommend the best option." },
  { intent: "cloud-vision", text: "What do you see in front of the camera?" },
  { intent: "cloud-vision", text: "Describe what is happening in this scene." },
  { intent: "cloud-vision", text: "Read the text on this sign I am showing you." },
  { intent: "cloud-vision", text: "What color is the shirt I am wearing?" },
  { intent: "cloud-vision", text: "How many fingers do you see?" },
  { intent: "cloud-vision", text: "Identify this object in my hand." },
  { intent: "cloud-vision", text: "Does this plant look healthy? Look at its leaves." },
  { intent: "cloud-vision", text: "Help me solve the problem written on this page." },
];

export function localDecision(reason: RoutingDecision["reason"]): RoutingDecision {
  return {
    intent: "transcript-only", transcriptionOnly: true,
    requiresCloud: false, requiresVision: false,
    similarity: 0, margin: 0, reason,
  };
}

export function transcriptGuard(text: string): RoutingDecision | null {
  if (!text.trim()) return localDecision("empty");
  if (text.trim().length > MAX_TRANSCRIPT_LENGTH) return localDecision("too-long");
  return null;
}

export function decideIntent(
  embedding: number[],
  examples: number[][],
): RoutingDecision {
  if (examples.length !== INTENT_EXAMPLES.length || !embedding.length ||
      !embedding.every(Number.isFinite) ||
      examples.some((example) => example.length !== embedding.length || !example.every(Number.isFinite))) {
    throw new Error("Invalid intent embeddings.");
  }
  const scores: Record<Intent, number> = {
    "transcript-only": -1, "cloud-text": -1, "cloud-vision": -1,
  };
  examples.forEach((example, index) => {
    const similarity = embedding.reduce((sum, value, dimension) => sum + value * example[dimension], 0);
    const intent = INTENT_EXAMPLES[index].intent;
    scores[intent] = Math.max(scores[intent], similarity);
  });
  const ranked = (Object.entries(scores) as [Intent, number][]).sort((left, right) => right[1] - left[1]);
  const [candidate, similarity] = ranked[0];
  const margin = similarity - ranked[1][1];
  if (similarity < MIN_SIMILARITY || margin < MIN_MARGIN) {
    return { ...localDecision("uncertain"), similarity, margin };
  }
  return {
    intent: candidate, transcriptionOnly: candidate === "transcript-only",
    requiresCloud: candidate !== "transcript-only", requiresVision: candidate === "cloud-vision",
    similarity, margin, reason: "matched",
  };
}