import React, { Component, ErrorInfo, ReactNode } from 'react';

interface Props {
  children: ReactNode;
  fallbackTitle?: string;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("ErrorBoundary caught an error:", error, errorInfo);
  }

  public render() {
    if (this.state.hasError) {
      return (
        <div className="p-6 my-4 bg-red-950/40 border border-red-800/60 rounded-xl text-red-200 shadow-lg max-w-xl mx-auto">
          <div className="flex items-center gap-3 mb-2">
            <span className="text-2xl">⚠️</span>
            <h3 className="font-semibold text-lg text-red-300">
              {this.props.fallbackTitle || "Rendering Error Detected"}
            </h3>
          </div>
          <p className="text-sm text-red-200/80 mb-4">
            An unexpected error occurred while rendering this view or 3D scene.
          </p>
          {this.state.error && (
            <pre className="p-3 bg-black/60 rounded border border-red-900/40 text-xs font-mono text-red-400 overflow-x-auto">
              {this.state.error.message}
            </pre>
          )}
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            className="mt-4 px-4 py-2 bg-red-800/60 hover:bg-red-700/80 text-white rounded-lg text-xs font-medium transition-colors"
          >
            Try Again
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
