"use client";

import { useState } from "react";

const navItems = [
  { name: "Overview", icon: "⊞", page: "home" },
  { name: "Live Map", icon: "🗺", page: "map" },
  { name: "Nearby Stations", icon: "⬡", page: "stations" },
  { name: "My Alerts", icon: "⚠", page: "alerts" },
];

const accountItems = [
  { name: "Profile", icon: "👤", page: "profile" },
  { name: "Settings", icon: "⚙", page: "settings" },
];

export default function Sidebar() {
  const [active, setActive] = useState("Overview");

  return (
    <div className="w-[240px] h-screen bg-white border-r flex flex-col p-4">

      {/* Logo */}
      <div className="flex items-center gap-2 pb-5 border-b mb-4">
        <div className="w-8 h-8 rounded-lg bg-blue-500 flex items-center justify-center text-white">
          💧
        </div>
        <span className="font-bold text-blue-900">AQUIS</span>
      </div>

      {/* User */}
      <div className="flex items-center gap-3 bg-gray-50 border rounded-md p-3 mb-5">
        <div className="w-8 h-8 rounded-full bg-purple-500 text-white flex items-center justify-center text-sm font-bold">
          SR
        </div>
        <div>
          <p className="text-sm font-semibold">Srijan Gupta</p>
          <p className="text-xs text-gray-400">General Public</p>
        </div>
      </div>

      {/* Navigation */}
      <div className="mb-4">
        <p className="text-xs text-gray-400 mb-2 px-1">NAVIGATION</p>

        {navItems.map((item) => (
          <div
            key={item.name}
            onClick={() => setActive(item.name)}
            className={`flex items-center gap-2 px-3 py-2 rounded-md cursor-pointer text-sm mb-1
              ${
                active === item.name
                  ? "bg-blue-100 text-blue-600 font-semibold"
                  : "text-gray-600 hover:bg-blue-50 hover:text-blue-600"
              }`}
          >
            <span>{item.icon}</span>
            {item.name}
          </div>
        ))}
      </div>

      {/* Account */}
      <div className="mb-4">
        <p className="text-xs text-gray-400 mb-2 px-1">ACCOUNT</p>

        {accountItems.map((item) => (
          <div
            key={item.name}
            onClick={() => setActive(item.name)}
            className={`flex items-center gap-2 px-3 py-2 rounded-md cursor-pointer text-sm mb-1
              ${
                active === item.name
                  ? "bg-blue-100 text-blue-600 font-semibold"
                  : "text-gray-600 hover:bg-blue-50 hover:text-blue-600"
              }`}
          >
            <span>{item.icon}</span>
            {item.name}
          </div>
        ))}
      </div>

      {/* Footer */}
      <div className="mt-auto pt-4 border-t">
        <div className="flex items-center gap-2 px-3 py-2 text-sm text-gray-400 hover:text-red-500 cursor-pointer">
          <span>↪</span> Logout
        </div>
      </div>
    </div>
  );
}