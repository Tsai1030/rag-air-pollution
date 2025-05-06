/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
      "./index.html",        // 依你的專案路徑調整
      "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
      extend: {
        colors: {
          pumpkin: '#D06A42',
          creamwhite: '#F9F8F3', // ← 
          creamlight: '#F2EFE7',
          milky: '#FCFBF9',
          cc:'#D38A74'
        },
      },
    },
    plugins: [],
  }
  
