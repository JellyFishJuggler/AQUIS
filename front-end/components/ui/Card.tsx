type Props = {
  children: React.ReactNode;
  className?: string;
};

export default function Card({ children, className = "" }: Props) {
  return (
    <div
      className={`bg-white border rounded-lg shadow-sm ${className}`}
    >
      {children}
    </div>
  );
}