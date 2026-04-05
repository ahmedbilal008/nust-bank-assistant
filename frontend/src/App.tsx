import { useState, useEffect, useRef } from "react"
import { motion } from "framer-motion"
import { HugeiconsIcon } from "@hugeicons/react"
import {
  Moon02Icon,
  Sun03Icon,
  ArrowUp02Icon,
  CloudUploadIcon,
} from "@hugeicons/core-free-icons"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Toaster } from "@/components/ui/sonner"
import { IngestDialog } from "@/components/IngestDialog"
import { cn } from "@/lib/utils"

const quickActions = [
  "Analyze my spending",
  "Review current portfolio",
  "Set a savings goal",
]

function App() {
  const [isDark, setIsDark] = useState(() => {
    if (typeof window !== "undefined") {
      return document.documentElement.classList.contains("dark")
    }
    return false
  })

  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add("dark")
    } else {
      document.documentElement.classList.remove("dark")
    }
  }, [isDark])

  const [query, setQuery] = useState("")
  const [messages, setMessages] = useState<{ role: "user" | "assistant"; content: string }[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [ingestOpen, setIngestOpen] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSend = async (text?: string) => {
    const message = text ?? query.trim()
    if (!message || isLoading) return

    setMessages(prev => [...prev, { role: "user", content: message }])
    setQuery("")
    setIsLoading(true)

    try {
      const res = await fetch("https://curblike-theologically-lavelle.ngrok-free.dev/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json", "ngrok-skip-browser-warning": "69420" },
        body: JSON.stringify({ query: message }),
      })

      if (!res.ok) {
        throw new Error(`Server error: ${res.status}`)
      }

      const data = await res.json()
      setMessages(prev => [...prev, { role: "assistant", content: data.answer }])
    } catch (err) {
      setMessages(prev => [...prev, { role: "assistant", content: "Sorry, something went wrong. Please try again." }])
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="relative min-h-screen overflow-hidden bg-[radial-gradient(1000px_500px_at_10%_0%,hsl(214_97%_27%/0.18),transparent),linear-gradient(160deg,#f5fbff_0%,#ebf3ff_55%,#f8fcff_100%)] text-slate-900 dark:bg-[radial-gradient(1000px_500px_at_10%_0%,hsl(214_97%_73%/0.18),transparent),linear-gradient(160deg,#0a1628_0%,#0d1f3c_55%,#0a1628_100%)] dark:text-slate-100">
      <motion.div
        className="pointer-events-none absolute inset-0"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.9 }}
      >
        <div className="absolute -top-20 left-[30%] size-72 rounded-full bg-primary/8 blur-3xl" />
        <div className="absolute bottom-0 right-0 size-96 rounded-full bg-sky-300/20 blur-3xl dark:bg-sky-500/10" />
      </motion.div>

      <motion.div
        className="relative mx-auto flex min-h-screen w-full flex-col sm:p-4"
      >
        <motion.div
          initial={false}
          transition={{ type: "spring", stiffness: 210, damping: 28 }}
          className="flex min-h-[92vh] flex-1 flex-col rounded-[34px] border border-white/70 bg-white/65 shadow-[0_18px_70px_-35px_hsl(214_97%_27%/0.7)] backdrop-blur-xl dark:border-white/10 dark:bg-slate-900/65 dark:shadow-[0_18px_70px_-35px_hsl(214_97%_73%/0.3)]"
        >
          <motion.header
            className="flex items-center justify-between px-4 py-4 sm:px-7"
            initial={{ y: -16, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.45 }}
          >
            <div className="flex items-center gap-3">
              <p className="font-robit text-base font-bold text-primary">NUST Bank</p>
            </div>

            

            <div className="flex items-center gap-2 sm:gap-3">
              <Button
                variant="ghost"
                size="icon-sm"
                className="rounded-full text-primary hover:bg-primary/8"
                onClick={() => setIsDark(!isDark)}
              >
                <HugeiconsIcon icon={isDark ? Sun03Icon : Moon02Icon} size={17} />
              </Button>
            </div>
          </motion.header>

          <main className="relative flex flex-1 flex-col px-4 pb-6 pt-8 sm:px-8 sm:pt-10">
            <motion.section
              className="mx-auto flex w-full max-w-4xl flex-1 flex-col items-center"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.25 }}
              transition={{ duration: 0.55, ease: [0.16, 1, 0.3, 1] }}
            >
              {/* Messages area */}
              {messages.length > 0 ? (
                <ScrollArea className="mb-6 flex-1 w-full">
                  <div className="space-y-4 pb-4">
                    {messages.map((msg, i) => (
                      <motion.div
                        key={i}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.3 }}
                        className={cn(
                          "flex w-full",
                          msg.role === "user" ? "justify-end" : "justify-start"
                        )}
                      >
                        <div
                          className={cn(
                            "max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed",
                            msg.role === "user"
                              ? "bg-primary text-primary-foreground"
                              : "bg-primary/8 text-slate-800 dark:bg-primary/15 dark:text-slate-200"
                          )}
                        >
                          {msg.content}
                        </div>
                      </motion.div>
                    ))}
                    {isLoading && (
                      <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="flex justify-start"
                      >
                        <div className="flex items-center gap-2 rounded-2xl bg-primary/8 px-4 py-3 text-sm text-slate-500 dark:bg-primary/15 dark:text-slate-400">
                          <span className="inline-flex gap-1">
                            <span className="size-1.5 animate-bounce rounded-full bg-primary/50" style={{ animationDelay: "0ms" }} />
                            <span className="size-1.5 animate-bounce rounded-full bg-primary/50" style={{ animationDelay: "150ms" }} />
                            <span className="size-1.5 animate-bounce rounded-full bg-primary/50" style={{ animationDelay: "300ms" }} />
                          </span>
                          Thinking...
                        </div>
                      </motion.div>
                    )}
                    <div ref={messagesEndRef} />
                  </div>
                </ScrollArea>
              ) : (
                <>
                  <Badge className="mb-6 rounded-full bg-primary/9 px-4 py-1.5 text-primary">Premium Banking Intelligence</Badge>
                  <h1 className="font-robit text-center text-4xl leading-tight font-bold text-primary sm:text-5xl">
                    NUST Bank Assistant
                  </h1>
                  <p className="mt-5 max-w-2xl text-center text-base leading-relaxed text-slate-600 sm:text-xl dark:text-slate-400">
                    Experience the future of precision banking. Securely manage assets, analyze spending, and forecast wealth with your dedicated AI financial partner.
                  </p>
                </>
              )}

              <motion.div
                className={cn("w-full", messages.length === 0 ? "mt-10" : "mt-auto")}
                initial={{ opacity: 0, scale: 0.97 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true, amount: 0.45 }}
                transition={{ delay: 0.12, duration: 0.45 }}
              >
                <div className="overflow-hidden rounded-[24px] border-[1.5px] border-primary/15 bg-gradient-to-b from-white/90 to-white/70 shadow-[0_24px_60px_-20px_hsl(214_97%_27%/0.25),inset_0_2px_16px_rgba(255,255,255,0.9)] backdrop-blur-2xl dark:from-slate-800/90 dark:to-slate-800/70 dark:shadow-[0_24px_60px_-20px_hsl(214_97%_73%/0.15),inset_0_2px_16px_rgba(255,255,255,0.05)]">
                  

                  {/* Textarea */}
                  <div className="px-5 py-2">
                    <Textarea
                      className="min-h-[48px] w-full resize-none border-0 bg-transparent p-0 text-[15px] leading-relaxed text-slate-800 placeholder:text-slate-400 focus-visible:ring-0 dark:text-slate-200 dark:placeholder:text-slate-500 pt-4"
                      placeholder="Ask about your balance, investments, or market trends..."
                      rows={1}
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      onKeyDown={handleKeyDown}
                    />
                  </div>

                  {/* Bottom toolbar */}
                  <div className="flex items-center justify-between border-t border-primary/10 px-5 py-2.5">
                    <div className="flex items-center gap-3">
                      <button
                        onClick={() => setIngestOpen(true)}
                        className="flex items-center gap-1.5 rounded-full border border-primary/15 bg-primary/5 px-3 py-1.5 text-xs font-medium text-primary/70 transition hover:bg-primary/10 hover:text-primary"
                      >
                        <HugeiconsIcon icon={CloudUploadIcon} size={13} strokeWidth={2} />
                        Upload Docs / FAQ / Text
                      </button>
                    </div>
                    <Button
                      className={cn(
                        "size-10 shrink-0 rounded-full border border-primary/15 p-0 shadow-lg transition",
                        query.trim()
                          ? "bg-primary text-primary-foreground shadow-primary/25 hover:bg-primary/90 cursor-pointer"
                          : "bg-primary/30 text-primary-foreground/50 shadow-none cursor-not-allowed"
                      )}
                      disabled={!query.trim() || isLoading}
                      onClick={() => handleSend()}
                    >
                      <HugeiconsIcon icon={ArrowUp02Icon} size={20} strokeWidth={2.2} />
                    </Button>
                  </div>
                </div>
              </motion.div>

              {messages.length === 0 && (
              <div className="mt-6 flex w-full flex-wrap items-center justify-center gap-2.5">
                {quickActions.map((action, index) => (
                  <motion.div
                    key={action}
                    initial={{ opacity: 0, y: 10 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true, amount: 0.5 }}
                    transition={{ duration: 0.33, delay: 0.24 + index * 0.07 }}
                  >
                    <Button
                      variant="outline"
                      className="h-10 rounded-full border-primary/12 bg-white/70 px-4 text-xs text-primary hover:bg-primary/7 dark:bg-slate-800/70 dark:hover:bg-primary/10"
                      onClick={() => handleSend(action)}
                    >
                      {`"${action}"`}
                    </Button>
                  </motion.div>
                ))}
              </div>
              )}

              
            </motion.section>
          </main>
        </motion.div>
      </motion.div>

      <IngestDialog open={ingestOpen} onOpenChange={setIngestOpen} />
      <Toaster position="bottom-right" theme={isDark ? "dark" : "light"} richColors />
    </div>
  )
}

export default App
