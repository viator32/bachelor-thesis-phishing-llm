import { useEffect, useMemo, useState } from "react";

// Render sortable, paginated record rows.
export default function SortableTable({ columns, rows, pageSize = 15 }) {
  const [sort, setSort] = useState({ key: null, dir: 1 });
  const [page, setPage] = useState(0);

  const sorted = useMemo(() => {
    if (!sort.key) return rows;
    return [...rows].sort((a, b) => {
      const av = a[sort.key];
      const bv = b[sort.key];
      if (av == null) return 1;
      if (bv == null) return -1;
      return av < bv ? -sort.dir : av > bv ? sort.dir : 0;
    });
  }, [rows, sort]);

  const pageCount = Math.max(1, Math.ceil(sorted.length / pageSize));
  // A narrower column set or a new run can leave the view past the last page.
  useEffect(() => {
    setPage((p) => Math.min(p, pageCount - 1));
  }, [pageCount]);

  const pageRows = sorted.slice(page * pageSize, (page + 1) * pageSize);
  // Check whether any row contains a number in a column.
  const isNumeric = (c) => rows.some((r) => typeof r[c] === "number");

  // Select or reverse column sorting and reset pagination.
  const toggleSort = (key) => {
    setSort((s) => (s.key === key ? { key, dir: -s.dir } : { key, dir: 1 }));
    setPage(0);
  };

  return (
    <div>
      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr>
              {columns.map((c) => (
                <th
                  key={c}
                  className={isNumeric(c) ? "num" : undefined}
                  onClick={() => toggleSort(c)}
                  aria-sort={sort.key === c ? (sort.dir === 1 ? "ascending" : "descending") : "none"}
                  style={{ cursor: "pointer" }}
                >
                  {c}
                  {sort.key === c ? (sort.dir === 1 ? " ▲" : " ▼") : ""}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {pageRows.map((row, i) => (
              <tr key={row.instance_id ?? i}>
                {columns.map((c) => (
                  <td key={c} className={isNumeric(c) ? "num" : undefined}>
                    {typeof row[c] === "number" ? Number(row[c].toFixed(3)) : String(row[c] ?? "")}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="table-tools" style={{ marginTop: "var(--s2)" }}>
        <button disabled={page === 0} onClick={() => setPage((p) => p - 1)}>
          Prev
        </button>
        <span>
          Page {page + 1} / {pageCount} · {sorted.length} records · {columns.length} columns
        </span>
        <button disabled={page >= pageCount - 1} onClick={() => setPage((p) => p + 1)}>
          Next
        </button>
      </div>
    </div>
  );
}
