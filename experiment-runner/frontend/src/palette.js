export const CATEGORICAL = [
  "#2a78d6",
  "#eb6834",
  "#1baf7a",
  "#eda100",
  "#e87ba4",
  "#008300",
  "#4a3aa7",
  "#e34948",
];
export const SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95"];
export const STATUS_GOOD = "#0ca30c";
export const STATUS_CRITICAL = "#d03b3b";
export const GRIDLINE = "#e1e0d9";
export const AXIS = "#c3c2b7";
export const INK_SECONDARY = "#52514e";
export const SURFACE = "#fcfcfb";

// Select a palette color by cycling through the palette.
export function colorAt(index) {
  return CATEGORICAL[index % CATEGORICAL.length];
}
