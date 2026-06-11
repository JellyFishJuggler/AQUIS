"use client";

type Props = {
  onChangeLocation?: () => void;
  onAnalytics?: () => void;
  onShare?: () => void;
  onProfile?: () => void;
};

export default function QuickActions({
  onChangeLocation,
  onAnalytics,
  onShare,
  onProfile,
}: Props) {
  const actions = [
    {
      label: "Change Location",
      icon: "📍",
      bg: "bg-blue-100",
      action: onChangeLocation,
    },
    {
      label: "View Analytics",
      icon: "📊",
      bg: "bg-green-100",
      action: onAnalytics,
    },
    {
      label: "Share Status",
      icon: "⇪",
      bg: "bg-purple-100",
      action: onShare,
    },
    {
      label: "My Profile",
      icon: "👤",
      bg: "bg-yellow-100",
      action: onProfile,
    },
  ];

  return (
    <div className="bg-white border rounded-lg">
      {/* Header */}
      <div className="px-4 py-3 border-b font-semibold text-sm">
        Quick Actions
      </div>

      {/* Grid */}
      <div className="grid grid-cols-2 gap-3 p-4">
        {actions.map((item) => (
          <div
            key={item.label}
            onClick={item.action}
            className="flex flex-col items-center justify-center gap-2 p-4 rounded-md border bg-gray-50 cursor-pointer hover:bg-blue-50 hover:border-blue-400 transition"
          >
            <div
              className={`w-10 h-10 flex items-center justify-center rounded-md text-lg ${item.bg}`}
            >
              {item.icon}
            </div>
            <span className="text-sm font-medium text-gray-700 text-center">
              {item.label}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}