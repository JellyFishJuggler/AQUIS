import Image from "next/image";

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-between p-24">
      <h1 className="text-4xl font-bold">Welcome to the Front-End!</h1>
      <p className="mt-4 text-lg">This is the home page of the front-end application.</p>
    </div>
  );
}
