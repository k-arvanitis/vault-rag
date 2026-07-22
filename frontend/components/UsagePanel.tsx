"use client";

import { useEffect, useState } from "react";
import { X } from "lucide-react";
import { getUsage, type UsageStats } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableRow } from "@/components/ui/table";

interface Props {
  onClose: () => void;
}

function formatCost(cost: number): string {
  return cost < 0.01 && cost > 0 ? "<$0.01" : `$${cost.toFixed(2)}`;
}

function formatLatency(ms: number | null): string {
  if (ms == null) return "—";
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`;
}

export default function UsagePanel({ onClose }: Props) {
  const [data, setData] = useState<UsageStats | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getUsage()
      .then((d) => {
        setData(d);
        setError(null);
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load usage"));
  }, []);

  const today = data?.daily[0];

  return (
    <div className="flex h-full w-full flex-col bg-background">
      <div className="flex shrink-0 items-center gap-3 border-b border-border bg-card px-5 py-3">
        <div className="min-w-0 flex-1">
          <p className="text-xs font-semibold text-foreground">LLM usage</p>
          <p className="text-[10px] text-muted-foreground">
            {data
              ? `${data.total_questions} questions tracked · ${formatCost(data.total_cost_usd)} all-time`
              : "Loading…"}
          </p>
        </div>
        <Button variant="ghost" size="icon-sm" onClick={onClose} aria-label="Close usage">
          <X />
        </Button>
      </div>

      <div className="flex-1 overflow-y-auto p-5">
        {error && (
          <Alert variant="destructive" className="mx-auto mb-3 max-w-3xl">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
        {!error && !data && (
          <div className="mx-auto max-w-3xl space-y-2">
            <Skeleton className="h-20 w-full" />
            <Skeleton className="h-40 w-full" />
          </div>
        )}
        {data && (
          <div className="mx-auto max-w-3xl space-y-5">
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <Card size="sm">
                <CardHeader>
                  <CardTitle>Today</CardTitle>
                </CardHeader>
                <CardContent className="text-xl font-semibold">
                  {today?.questions ?? 0}
                  <span className="ml-1 text-xs font-normal text-muted-foreground">questions</span>
                </CardContent>
              </Card>
              <Card size="sm">
                <CardHeader>
                  <CardTitle>Today tokens</CardTitle>
                </CardHeader>
                <CardContent className="text-xl font-semibold">
                  {(today?.total_tokens ?? 0).toLocaleString()}
                </CardContent>
              </Card>
              <Card size="sm">
                <CardHeader>
                  <CardTitle>Today cost (est.)</CardTitle>
                </CardHeader>
                <CardContent className="text-xl font-semibold">
                  {formatCost(today?.cost_usd ?? 0)}
                </CardContent>
              </Card>
              <Card size="sm">
                <CardHeader>
                  <CardTitle>Today avg latency</CardTitle>
                </CardHeader>
                <CardContent className="text-xl font-semibold">
                  {formatLatency(today?.avg_latency_ms ?? null)}
                </CardContent>
              </Card>
            </div>

            <div>
              <p className="mb-2 text-xs font-medium text-foreground">Daily usage</p>
              {data.daily.length === 0 ? (
                <p className="text-sm text-muted-foreground">No usage recorded yet.</p>
              ) : (
                <div className="overflow-hidden rounded-lg border border-border">
                  <Table>
                    <TableBody>
                      {data.daily.map((d) => (
                        <TableRow key={d.date}>
                          <TableCell className="text-foreground">{d.date}</TableCell>
                          <TableCell className="text-right text-muted-foreground">
                            {d.questions} question{d.questions === 1 ? "" : "s"}
                          </TableCell>
                          <TableCell className="text-right text-muted-foreground">
                            {d.total_tokens.toLocaleString()} tok
                          </TableCell>
                          <TableCell className="w-20 text-right text-muted-foreground">
                            {formatLatency(d.avg_latency_ms)} avg
                          </TableCell>
                          <TableCell className="w-20 text-right text-foreground">
                            {formatCost(d.cost_usd)}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              )}
            </div>

            <div>
              <p className="mb-2 text-xs font-medium text-foreground">Recent questions</p>
              {data.recent.length === 0 ? (
                <p className="text-sm text-muted-foreground">No questions logged yet.</p>
              ) : (
                <div className="overflow-hidden rounded-lg border border-border">
                  <Table>
                    <TableBody>
                      {data.recent.map((e, i) => (
                        <TableRow key={`${e.timestamp}-${i}`}>
                          <TableCell className="max-w-0 truncate whitespace-nowrap text-foreground">
                            {e.question}
                          </TableCell>
                          <TableCell className="w-28 truncate text-[10px] text-muted-foreground">
                            {e.model ?? "unknown"}
                          </TableCell>
                          <TableCell className="w-24 text-right text-muted-foreground">
                            {e.total_tokens.toLocaleString()} tok
                          </TableCell>
                          <TableCell className="w-16 text-right text-muted-foreground">
                            {formatLatency(e.latency_ms)}
                          </TableCell>
                          <TableCell className="w-16 text-right text-foreground">
                            {formatCost(e.cost_usd)}
                          </TableCell>
                          <TableCell className="w-36 text-right text-[10px] text-muted-foreground">
                            {new Date(e.timestamp).toLocaleString()}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
