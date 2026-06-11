import StatCard from "../ui/StatCard";

type Props = {
  district?: string;
  state?: string;
  extraction?: number;
  rainfall?: number;
  recharge?: number;
  extractionTotal?: number;
};

export default function StatGrid({
  district = "—",
  state = "—",
  extraction = 0,
  rainfall = 0,
  recharge = 0,
  extractionTotal = 0,
}: Props) {
  return (
    <div className="grid grid-cols-4 gap-4 mt-4">

      {/* HERO CARD */}
      <StatCard
        title={`Extraction Rate · ${district}`}
        value={`${extraction}%`}
        sub={state}
        highlight
      />

      {/* Rainfall */}
      <StatCard
        title="Rainfall"
        value={`${rainfall} mm`}
      />

      {/* Recharge */}
      <StatCard
        title="Recharge"
        value={`${(recharge / 1000).toFixed(1)}k`}
        sub="hectare-metres"
      />

      {/* Extraction */}
      <StatCard
        title="Total Extraction"
        value={`${(extractionTotal / 1000).toFixed(1)}k`}
        sub="hectare-metres"
      />
    </div>
  );
}