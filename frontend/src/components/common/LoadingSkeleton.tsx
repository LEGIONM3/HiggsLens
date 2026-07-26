import React from 'react';

interface LoadingSkeletonProps {
  height?: string;
  width?: string;
  className?: string;
}

export const LoadingSkeleton: React.FC<LoadingSkeletonProps> = ({
  height = 'h-24',
  width = 'w-full',
  className = '',
}) => {
  return (
    <div
      aria-label="Loading component content..."
      className={`animate-pulse bg-slate-800/60 rounded-xl border border-slate-700/50 ${height} ${width} ${className}`}
    />
  );
};
