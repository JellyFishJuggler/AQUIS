"use client";

import React from "react";

type Props = {
  open: boolean;
  onClose: () => void;
  title?: string;
  children: React.ReactNode;
};

export default function Modal({ open, onClose, title, children }: Props) {
  if (!open) return null;

  return (
    <div
      className="fixed inset-0 bg-black/50 flex items-center justify-center z-50"
      onClick={onClose} // click outside closes
    >
      {/* Modal Box */}
      <div
        className="bg-white rounded-lg shadow-lg w-[400px] max-w-[90%] p-5"
        onClick={(e) => e.stopPropagation()} // prevent close on inside click
      >
        {/* Header */}
        {title && (
          <div className="text-lg font-semibold mb-2">{title}</div>
        )}

        {/* Content */}
        <div className="text-sm text-gray-600">
          {children}
        </div>

        {/* Footer */}
        <div className="flex justify-end gap-2 mt-4">
          <button
            onClick={onClose}
            className="px-3 py-1 rounded-md border text-sm"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}