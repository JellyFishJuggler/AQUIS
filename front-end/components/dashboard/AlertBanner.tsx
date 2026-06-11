"use client";

type Status = "SAFE" | "SEMI_CRITICAL" | "CRITICAL" | "OVER_EXPLOITED";

type Props = {
  status: Status;
};

export default function AlertBanner({ status }: Props) {
  const config = {
    SAFE: {
      text: "✅ Groundwater levels are safe.",
      style: "bg-green-100 text-green-700",
    },
    SEMI_CRITICAL: {
      text: "⚠️ Approaching semi-critical levels. Use water mindfully.",
      style: "bg-yellow-100 text-yellow-700",
    },
    CRITICAL: {
      text: "🔶 Critical groundwater stress detected. Conservation needed.",
      style: "bg-orange-100 text-orange-700",
    },
    OVER_EXPLOITED: {
      text: "🔴 Groundwater is over-exploited. Immediate action required.",
      style: "bg-red-100 text-red-700",
    },
  };

  const { text, style } = config[status];

  return (
    <div
      className={`w-full p-3 rounded-md text-sm font-medium flex items-center gap-2 ${style}`}
    >
      {text}
    </div>
  );
}