import { NextResponse } from "next/server";

export async function POST(request: Request) {
  try {
    const { image, transcript, routing } = await request.json();
    if (typeof transcript !== "string" || !transcript.trim() || transcript.length > 1200 ||
        routing?.requiresCloud !== true || typeof routing.requiresVision !== "boolean") {
      return NextResponse.json({ success: false, error: "A transcript and affirmative cloud routing decision are required." }, { status: 400 });
    }
    if ((routing.requiresVision && (typeof image !== "string" || !/^data:image\/jpeg;base64,[A-Za-z0-9+/=]+$/.test(image) || image.length > 2_000_000)) ||
        (!routing.requiresVision && image !== undefined)) {
      return NextResponse.json({ success: false, error: "Image payload must match the visual routing decision." }, { status: 400 });
    }
    if (!process.env.NEBIUS_API_KEY) {
      return NextResponse.json({ success: false, error: "NEBIUS_API_KEY is not configured." }, { status: 503 });
    }
    if (!process.env.NEBIUS_ENDPOINT) {
      return NextResponse.json({ success: false, error: "NEBIUS_ENDPOINT is not configured." }, { status: 503 });
    }
    if (!process.env.NEBIUS_MODEL?.trim()) {
      return NextResponse.json({ success: false, error: "NEBIUS_MODEL is not configured." }, { status: 503 });
    }

    // Forward the image data securely to Nebius Token Factory
    const nebiusResponse = await fetch(`${process.env.NEBIUS_ENDPOINT.replace(/\/+$/, "")}/chat/completions`, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${process.env.NEBIUS_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: process.env.NEBIUS_MODEL,
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: transcript.trim() },
              ...(routing.requiresVision ? [{ type: "image_url", image_url: { url: image } }] : [])
            ]
          }
        ]
      }),
      signal: request.signal,
    });

    const aiData = await nebiusResponse.json();

    if (!nebiusResponse.ok) {
      return NextResponse.json(
        { success: false, error: aiData.error?.message ?? aiData.detail ?? "Nebius request failed" },
        { status: nebiusResponse.status }
      );
    }

    return NextResponse.json({ success: true, analysis: aiData });
  } catch (error) {
    // Log the error for debugging purposes
    console.error(error);

    const errorMessage = error instanceof Error ? error.message : "An unknown error occurred";
    return NextResponse.json({ success: false, error: errorMessage }, { status: 500 });
  }
}
