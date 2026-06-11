"use client";

import { useEffect, useRef } from "react";
import Chart from "chart.js/auto";

type Props = {
  extraction: number; // %
  recharge: number;
  extractionTotal: number;
};

export default function Charts({
  extraction,
  recharge,
  extractionTotal,
}: Props) {
  const donutRef = useRef<HTMLCanvasElement | null>(null);
  const barRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (!donutRef.current || !barRef.current) return;

    // Destroy old charts (important)
    Chart.getChart(donutRef.current)?.destroy();
    Chart.getChart(barRef.current)?.destroy();

    // 🎯 Donut Chart
    new Chart(donutRef.current, {
      type: "doughnut",
      data: {
        datasets: [
          {
            data: [extraction, 100 - extraction],
          },
        ],
      },
      options: {
        cutout: "70%",
        plugins: {
          legend: { display: false },
        },
      },
    });

    // 📊 Bar Chart
    new Chart(barRef.current, {
      type: "bar",
      data: {
        labels: ["Recharge", "Extraction"],
        datasets: [
          {
            data: [recharge, extractionTotal],
          },
        ],
      },
      options: {
        plugins: {
          legend: { display: false },
        },
      },
    });
  }, [extraction, recharge, extractionTotal]);

  return (
    <div className="grid grid-cols-2 gap-4 mt-4">

      {/* Donut */}
      <div className="bg-white border rounded-lg p-4">
        <p className="text-xs text-gray-400 mb-2">Usage vs Capacity</p>
        <canvas ref={donutRef} />
      </div>

      {/* Bar */}
      <div className="bg-white border rounded-lg p-4">
        <p className="text-xs text-gray-400 mb-2">Water Balance</p>
        <canvas ref={barRef} />
      </div>

    </div>
  );
}