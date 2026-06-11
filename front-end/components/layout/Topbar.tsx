"use client";

export default function Topbar() {
  return (
    <div className="h-[60px] bg-white border-b flex items-center justify-between px-6">

      {/* Left side */}
      <div>
        <h2 className="text-sm font-semibold text-gray-800">
          Hi, Srijan!
        </h2>
        <p className="text-xs text-gray-400">
          AQUIS Public Dashboard
        </p>
      </div>

      {/* Right side */}
      <div className="flex items-center gap-3">

        {/* Location chip */}
        <div className="flex items-center gap-1 px-3 py-1 rounded-full bg-gray-100 border text-sm text-gray-600">
          📍 <span>No location</span>
        </div>

        {/* Notification */}
        <div className="relative w-9 h-9 flex items-center justify-center rounded-full border bg-gray-100 cursor-pointer hover:bg-blue-50">
          🔔
          <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full border border-white"></span>
        </div>

      </div>
    </div>
  );
}