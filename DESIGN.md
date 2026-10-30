Please update the Tailwind CSS styling for the following filesto match the visual layout, spacing, and styling in docs/redesign-mockup.png:
- app/layout.tsx 
- app/page.tsx 
- app/components/AudioTranscriberStreaming.tsx
- app/components/AudioVisualizer.tsx

Instructions:
- Keep all existing Next.js 16 App Router logic, state, hooks, workers, components, and async data-fetching code completely intact.
- Rewrite the utility classes to match the design's spacing, Flexbox/Grid layouts, and alignment.
- Tailwind v4 Rules: Do not attempt to modify or create a tailwind.config.js file. If you need to extend the theme with custom brand colors or fonts extracted from the image, add them to app/globals.css inside the @theme directive (e.g., --color-brand-primary: #HEX;) and reference them via utility classes like bg-brand-primary.
