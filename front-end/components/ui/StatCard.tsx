type Props = {
  title: string;
  value: string;
  sub?: string;
  highlight?: boolean;
};

export default function StatCard({ title, value, sub, highlight }: Props) {
  return (
    <div
      className={`p-5 rounded-lg border bg-white ${
        highlight ? "bg-blue-600 text-white" : ""
      }`}
    >
      <p className="text-xs uppercase tracking-wide opacity-70 mb-2">
        {title}
      </p>

      <h2 className={`text-2xl font-bold ${highlight ? "" : "text-gray-800"}`}>
        {value}
      </h2>

      {sub && (
        <p className="text-xs mt-1 opacity-70">
          {sub}
        </p>
      )}
    </div>
  );
}