/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        observatory: {
          bg: '#05070c',
          panel: '#090d16',
          surface: '#0f172a',
          beam: '#06b6d4',
          lepton: '#f8fafc',
          jetLeading: '#f59e0b',
          jetSubleading: '#fbbf24',
          met: '#ec4899',
        },
      },
    },
  },
  plugins: [],
};
