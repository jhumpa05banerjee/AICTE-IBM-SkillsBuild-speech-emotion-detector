"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { ClassificationResult } from "@/lib/emotion-classifier";
import { getEmotionLabel } from "@/lib/emotion-classifier";

/**
 * EmotionResults
 *
 * Displays the predicted emotion probabilities from the classifier.
 * Shows:
 *  1. The dominant emotion with confidence percentage
 *  2. A horizontal bar chart of all emotion probabilities
 *  3. A detailed breakdown table
 */

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload || !payload.length) return null;

  return (
    <div className="bg-white border border-gray-300 rounded-lg px-3 py-2 shadow-lg">
      <p className="font-semibold text-black">{label}</p>
      <p className="text-sm text-gray-700">
        Probability: <span className="font-medium">{payload[0].value}%</span>
      </p>
    </div>
  );
};

interface EmotionResultsProps {
  result: ClassificationResult;
}

export function EmotionResults({ result }: EmotionResultsProps) {
  const { predictions, dominantEmotion, confidence } = result;

  // Prepare chart data
  const chartData = predictions.map((p) => ({
    name: getEmotionLabel(p.emotion),
    probability: Math.round(p.probability * 1000) / 10, // e.g. 45.3%
    fill: p.color,
  }));

  return (
    <div className="flex flex-col gap-6">
      {/* Model mode indicator */}
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <div
          className={`h-2 w-2 rounded-full ${
            result.mode === "model" ? "bg-emerald-500" : "bg-amber-500"
          }`}
        />
        {result.mode === "model" ? (
          <span>Using trained MLP model</span>
        ) : (
          <span>
            On your given audio, we will run our saved model to detect your emotions
          </span>
        )}
      </div>

      {/* Dominant emotion hero */}
      <Card className="border-primary/30 bg-card">
        <CardContent className="flex flex-col items-center gap-4 py-8">
          <div className="text-sm font-medium uppercase tracking-widest text-muted-foreground">
            Detected Emotion
          </div>

          {/* Emotion circle */}
          <div
            className="relative flex h-28 w-28 items-center justify-center rounded-full"
            style={{
              background: `radial-gradient(circle, ${predictions[0].color}22, transparent 70%)`,
              border: `2px solid ${predictions[0].color}66`,
            }}
          >
            <div className="text-center">
              <div
                className="text-2xl font-bold"
                style={{ color: predictions[0].color }}
              >
                {getEmotionLabel(dominantEmotion)}
              </div>
              <div className="text-xs text-muted-foreground mt-1">
                {(confidence * 100).toFixed(1)}%
              </div>
            </div>
          </div>

          <p className="text-sm text-muted-foreground text-center max-w-sm">
            The model detected{" "}
            <span
              className="font-semibold"
              style={{ color: predictions[0].color }}
            >
              {getEmotionLabel(dominantEmotion)}
            </span>{" "}
            as the primary emotion with{" "}
            {(confidence * 100).toFixed(1)}% confidence.
          </p>
        </CardContent>
      </Card>

      {/* Bar chart */}
      <Card className="bg-card">
        <CardHeader>
          <CardTitle className="text-base">Emotion Probabilities</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart
              data={chartData}
              layout="vertical"
              margin={{ top: 0, right: 30, left: 10, bottom: 0 }}
              barCategoryGap="20%"
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="rgba(148, 163, 184, 0.1)"
                horizontal={false}
              />
              <XAxis
                type="number"
                domain={[0, 100]}
                tickFormatter={(v) => `${v}%`}
                tick={{ fill: "rgba(4, 18, 36, 0.6)", fontSize: 12 }}
                axisLine={{ stroke: "rgba(148, 163, 184, 0.15)" }}
                tickLine={false}
              />
              <YAxis
                type="category"
                dataKey="name"
                tick={{ fill: "rgba(2, 11, 24, 0.8)", fontSize: 13 }}
                axisLine={false}
                tickLine={false}
                width={70}
              />
              <Tooltip content={<CustomTooltip />} cursor={{ fill: "#e5e7eb" }} />
              
              <Bar dataKey="probability" radius={[0, 6, 6, 0]} maxBarSize={28}>
                {chartData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.fill} fillOpacity={1} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Detailed breakdown */}
      <Card className="bg-card">
        <CardHeader>
          <CardTitle className="text-base">Detailed Breakdown</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col gap-3">
            {predictions.map((pred) => (
              <div key={pred.emotion} className="flex items-center gap-4">
                {/* Color dot */}
                <div
                  className="h-3 w-3 rounded-full shrink-0"
                  style={{ backgroundColor: pred.color }}
                />

                {/* Label */}
                <span className="text-sm font-medium text-foreground w-20">
                  {getEmotionLabel(pred.emotion)}
                </span>

                {/* Progress bar */}
                <div className="flex-1 h-2 rounded-full bg-secondary/50 overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-700 ease-out"
                    style={{
                      width: `${pred.probability * 100}%`,
                      backgroundColor: pred.color,
                      opacity: 1,
                    }}
                  />
                </div>

                {/* Percentage */}
                <span className="text-sm font-mono text-muted-foreground w-14 text-right">
                  {(pred.probability * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
