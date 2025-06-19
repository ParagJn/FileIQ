import React from 'react';

interface ProgressItem {
  file: string;
  progress: number; // 0-100
  status: 'pending' | 'processing' | 'done' | 'error' | 'skipped';
  message?: string;
}

interface TerminalProgressModalProps {
  open: boolean;
  onClose: () => void;
  items: ProgressItem[];
  title?: string;
}

export const TerminalProgressModal: React.FC<TerminalProgressModalProps> = ({ open, onClose, items, title = "Vector Generation Progress" }) => {
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="bg-[#18181b] rounded-lg shadow-2xl w-full max-w-xl border border-gray-700 relative">
        <button
          className="absolute top-2 right-2 text-gray-400 hover:text-red-400 text-xl font-bold"
          onClick={onClose}
          aria-label="Close"
        >
          ×
        </button>
        <div className="p-6 font-mono text-sm text-green-200">
          <div className="mb-4 text-lg text-green-400 font-bold">[Terminal] {title}</div>
          {items.map((item, idx) => (
            <div key={item.file} className="mb-4">
              <div className="flex items-center gap-2">
                <span className="text-blue-300">{idx + 1}.</span>
                <span className="truncate max-w-xs">{item.file}</span>
                <span className="ml-auto text-xs text-gray-400">
                  {item.status === 'done' ? 'Done' : 
                   item.status === 'error' ? 'Error' : 
                   item.status === 'skipped' ? 'Skipped' : 
                   'Processing...'}
                </span>
              </div>
              <div className="w-full bg-gray-800 rounded h-3 mt-1">
                <div
                  className={`h-3 rounded ${
                    item.status === 'error' ? 'bg-red-500' : 
                    item.status === 'skipped' ? 'bg-yellow-500' : 
                    'bg-green-500'
                  }`}
                  style={{ width: `${item.progress}%`, transition: 'width 0.3s' }}
                />
              </div>
              {item.message && (
                <div className={`mt-1 ${
                  item.status === 'error' ? 'text-red-400' : 
                  item.status === 'skipped' ? 'text-yellow-400' : 
                  'text-green-400'
                }`}>
                  {item.message}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
