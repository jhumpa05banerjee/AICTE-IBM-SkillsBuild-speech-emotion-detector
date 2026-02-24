import { SpeechEmotionDetector } from "@/components/speech-emotion-detector";

export default function Page() {
  return (
    <main className="min-h-screen bg-background">
      <div className="mx-auto max-w-2xl px-4 py-12 sm:px-6 lg:px-8">
        <SpeechEmotionDetector />

        {/* Footer */}
        <footer className="mt-16 border-t border-border/30 pt-6 text-center">
          <p className="text-xs text-muted-foreground">
            Speech Emotion Detection - Made with love, by Jhumpa Banerjee 
          </p>
        </footer>
      </div>
    </main>
  );
}
