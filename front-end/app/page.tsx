"use client";

import Image from "next/image";
import Charts from "@/components/dashboard/Charts";
import AlertBanner from "@/components/dashboard/AlertBanner";
import QuickActions from "@/components/dashboard/QuickActions";
import StatGrid from "@/components/dashboard/StatGrid";

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-between p-24">
      {/* <h1 className="text-4xl font-bold">Welcome to the Front-End!</h1>
      <p className="mt-4 text-lg">This is the home page of the front-end application.</p> */}
      <AlertBanner status="CRITICAL" />
      <StatGrid
        district="District Name"
        state="State Name"
        extraction={82}
        rainfall={100}
        recharge={40000}
        extractionTotal={33000}
      />
      <Charts
        extraction={82}
        recharge={40000}
        extractionTotal={33000}
      />
      <QuickActions
        onChangeLocation={() => console.log("change")}
        onAnalytics={() => console.log("analytics")}
        onShare={() => console.log("share")}
        onProfile={() => console.log("profile")}
      />
    </div>
  );
}
