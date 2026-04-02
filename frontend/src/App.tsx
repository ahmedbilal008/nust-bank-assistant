import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import { HugeiconsIcon } from "@hugeicons/react"
import {
  Add01Icon,
  Attachment01Icon,

  ChatBotIcon,
  ChatIcon,
  InternetIcon,
  Logout01Icon,
  Moon02Icon,
  Sun03Icon,
  ArrowUp02Icon,
} from "@hugeicons/core-free-icons"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { ScrollArea } from "@/components/ui/scroll-area"
import { cn } from "@/lib/utils"

import {
  SidebarProvider,
  SidebarTrigger,
  useSidebar,
} from "@/components/ui/sidebar"

const quickActions = [
  "Analyze my spending",
  "Review current portfolio",
  "Set a savings goal",
]

const historyItems = [
  "How can I optimize card usage this month?",
  "Should I increase my emergency fund?",
  "Estimate zakat for my current assets",
  "Where is my spending leakage?",
  "Create a low-risk investment path",
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

      <SidebarProvider>
        <AppLayout isDark={isDark} toggleDark={() => setIsDark(!isDark)} />
      </SidebarProvider>
    </div>
  )
}

function AppLayout({ isDark, toggleDark }: { isDark: boolean; toggleDark: () => void }) {
  const { state } = useSidebar()
  const isCollapsed = state === "collapsed"

  return (
    <motion.div layout className="relative mx-auto flex min-h-screen w-full sm:p-4">
      <motion.aside
        layout
        initial={false}
        animate={{ width: isCollapsed ? 96 : 340 }}
        transition={{ type: "spring", stiffness: 210, damping: 28 }}
        className="hidden shrink-0 overflow-hidden lg:block lg:pr-5"
      >
            <Card className="h-full rounded-[34px] border-primary/10 bg-white/78 shadow-[0_16px_60px_-30px_hsl(214_97%_27%/0.6)] backdrop-blur-xl dark:bg-slate-900/78 dark:shadow-[0_16px_60px_-30px_hsl(214_97%_73%/0.3)]">
              <CardContent className={cn("flex h-full flex-col", isCollapsed ? "p-2" : "p-5")}>
                <div className={cn("mb-6 flex items-center", isCollapsed ? "justify-center" : "justify-between")}>
                  <div className="flex items-center gap-3">
                    
                    {!isCollapsed ? (
                      <div>
                      <p className="font-robit whitespace-nowrap text-lg font-bold text-primary">NUST Bank</p>
                      <p className="whitespace-nowrap text-xs text-slate-500 dark:text-slate-400">Precision Banking</p>
                      </div>
                    ) : null}
                  </div>
                  <SidebarTrigger className={cn(isCollapsed && "rounded-full")} />
                </div>

                <div className="flex h-full flex-col">
                  <Button variant="ghost" className={cn("h-12 w-full gap-2 rounded-2xl px-4 text-sm font-semibold text-primary hover:bg-primary/8", isCollapsed ? "justify-center" : "justify-start")}>
                    <HugeiconsIcon icon={Add01Icon} size={22} strokeWidth={2.5} />
                    {!isCollapsed ? "New Chat" : null}
                  </Button>

                  <ScrollArea className={cn("h-[58vh] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden", isCollapsed ? "pr-0" : "pr-2")}>
                    

                  

                  {!isCollapsed ? <div className="space-y-2">
                    {historyItems.map((item, index) => (
                      <motion.div
                        key={item}
                        initial={{ opacity: 0, x: -12 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.08 * index + 0.2, duration: 0.35 }}
                      >
                        <Button
                          variant="ghost"
                          className="h-auto w-full justify-start rounded-2xl px-3.5 py-3 text-left text-sm leading-relaxed text-slate-500 hover:bg-primary/5 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200"
                        >
                          {item}
                        </Button>
                      </motion.div>
                    ))}
                  </div> : null}
                </ScrollArea>

                <div className={cn("mt-auto space-y-2 pt-4", isCollapsed && "flex flex-col items-center")}>
                 
                    <Button variant="ghost" className={cn("h-11 w-full gap-2.5 px-3.5 text-slate-600 hover:bg-rose-50 hover:text-rose-500 dark:text-slate-400 dark:hover:bg-rose-950/30", isCollapsed ? "justify-center rounded-full" : "justify-start rounded-2xl")}>
                      <HugeiconsIcon icon={Logout01Icon} size={17} strokeWidth={1.9} />
                      {!isCollapsed ? "Log Out" : null}
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
      </motion.aside>

      <motion.div
        layout
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
            <div className="flex items-center gap-3 lg:hidden">
              <Button variant="outline" size="icon-sm" className="rounded-xl border-primary/20 bg-white/90 text-primary dark:bg-slate-800/90">
                <HugeiconsIcon icon={ChatIcon} size={16} />
              </Button>
              <p className="font-robit text-base font-bold text-primary">NUST Bank</p>
            </div>

            <div className="hidden items-center gap-2 lg:flex">
              <Badge variant="secondary" className="rounded-full px-3 py-1 text-[11px] font-semibold">Assistant</Badge>
              
            </div>

            <div className="flex items-center gap-2 sm:gap-3">
              <Button
                variant="ghost"
                size="icon-sm"
                className="rounded-full text-primary hover:bg-primary/8"
                onClick={toggleDark}
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
              <Badge className="mb-6 rounded-full bg-primary/9 px-4 py-1.5 text-primary">Premium Banking Intelligence</Badge>
              <h1 className="font-robit text-center text-4xl leading-tight font-bold text-primary sm:text-5xl">
                NUST Bank Assistant
              </h1>
              <p className="mt-5 max-w-2xl text-center text-base leading-relaxed text-slate-600 sm:text-xl dark:text-slate-400">
                Experience the future of precision banking. Securely manage assets, analyze spending, and forecast wealth with your dedicated AI financial partner.
              </p>

              <motion.div
                className="mt-10 w-full"
                initial={{ opacity: 0, scale: 0.97 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true, amount: 0.45 }}
                transition={{ delay: 0.12, duration: 0.45 }}
              >
                <div className="overflow-hidden rounded-[24px] border-[1.5px] border-primary/15 bg-gradient-to-b from-white/90 to-white/70 shadow-[0_24px_60px_-20px_hsl(214_97%_27%/0.25),inset_0_2px_16px_rgba(255,255,255,0.9)] backdrop-blur-2xl dark:from-slate-800/90 dark:to-slate-800/70 dark:shadow-[0_24px_60px_-20px_hsl(214_97%_73%/0.15),inset_0_2px_16px_rgba(255,255,255,0.05)]">
                  {/* Top row — @Add context pill */}
                  <div className="px-5 pt-4 pb-1">
                    <button className="inline-flex items-center gap-1.5 rounded-full border border-primary/15 bg-primary/[0.06] px-3.5 py-1.5 text-xs font-medium text-primary/70 transition hover:border-primary/30 hover:bg-primary/10 hover:text-primary">
                      <span className="text-sm font-semibold leading-none">@</span>
                      Add context
                    </button>
                  </div>

                  {/* Textarea */}
                  <div className="px-5 py-2">
                    <Textarea
                      className="min-h-[48px] w-full resize-none border-0 bg-transparent p-0 text-[15px] leading-relaxed text-slate-800 placeholder:text-slate-400 focus-visible:ring-0 dark:text-slate-200 dark:placeholder:text-slate-500"
                      placeholder="Ask about your balance, investments, or market trends..."
                      rows={1}
                    />
                  </div>

                  {/* Bottom toolbar */}
                  <div className="flex items-center justify-between border-t border-primary/10 px-5 py-2.5">
                    <div className="flex items-center gap-4">
                      <button className="text-primary/40 transition hover:text-primary/70">
                        <HugeiconsIcon icon={Attachment01Icon} size={18} strokeWidth={1.8} />
                      </button>
                      <span className="text-xs font-medium text-primary/50">Auto</span>
                      <button className="inline-flex items-center gap-1.5 text-xs font-medium text-primary/50 transition hover:text-primary/80">
                        <HugeiconsIcon icon={InternetIcon} size={16} strokeWidth={1.8} />
                        All Sources
                      </button>
                    </div>
                    <Button className="size-10 shrink-0 rounded-full border border-primary/15 bg-primary p-0 text-primary-foreground shadow-lg shadow-primary/25 transition hover:bg-primary/90">
                      <HugeiconsIcon icon={ArrowUp02Icon} size={20} strokeWidth={2.2} />
                    </Button>
                  </div>
                </div>
              </motion.div>

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
                    >
                      {`\"${action}\"`}
                    </Button>
                  </motion.div>
                ))}
              </div>

              
            </motion.section>
          </main>
      </motion.div>
    </motion.div>
  )
}

export default App
