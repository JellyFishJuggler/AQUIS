"use client";

import React from "react";

type ButtonProps = {
  children: React.ReactNode;
  onClick?: () => void;
  variant?: "primary" | "secondary" | "ghost" | "danger";
  className?: string;
};

export default function Button({
  children,
  onClick,
  variant = "primary",
  className = "",
}: ButtonProps) {
  const base =
    "px-4 py-2 rounded-md text-sm font-semibold transition-all duration-150";

  const variants = {
    primary: "bg-blue-600 text-white hover:bg-blue-700",
    secondary: "bg-gray-200 text-gray-800 hover:bg-gray-300",
    ghost: "bg-transparent text-gray-600 hover:bg-gray-100",
    danger: "bg-red-500 text-white hover:bg-red-600",
  };

  return (
    <button
      onClick={onClick}
      className={`${base} ${variants[variant]} ${className}`}
    >
      {children}
    </button>
  );
}