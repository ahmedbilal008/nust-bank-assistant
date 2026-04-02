import { motion } from "framer-motion"
import { HugeiconsIcon } from "@hugeicons/react"
import {
  ArrowRight01Icon,
  ChatAdd01Icon,
  ChatBotIcon,
  ChatIcon,
  HelpCircleIcon,
  Logout01Icon,
  Notification01Icon,
  Search01Icon,
} from "@hugeicons/core-free-icons"

import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
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
  return (
    <div className="relative min-h-screen overflow-hidden bg-[radial-gradient(1000px_500px_at_10%_0%,rgba(2,62,138,0.18),transparent),linear-gradient(160deg,#f5fbff_0%,#ebf3ff_55%,#f8fcff_100%)] text-slate-900">
      <motion.div
        className="pointer-events-none absolute inset-0"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.9 }}
      >
        <div className="absolute -top-20 left-[30%] size-72 rounded-full bg-[#023e8a]/8 blur-3xl" />
        <div className="absolute bottom-0 right-0 size-96 rounded-full bg-sky-300/20 blur-3xl" />
      </motion.div>

      <SidebarProvider>
        <AppLayout />
      </SidebarProvider>
    </div>
  )
}

function AppLayout() {
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
            <Card className="h-full rounded-[34px] border-[#023e8a]/10 bg-white/78 shadow-[0_16px_60px_-30px_rgba(2,62,138,0.6)] backdrop-blur-xl">
              <CardContent className="flex h-full flex-col p-5">
                <div className="mb-6 flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="grid size-11 shrink-0 place-content-center rounded-2xl bg-[#023e8a] text-white">
                      <HugeiconsIcon icon={ChatBotIcon} size={22} strokeWidth={1.7} />
                    </div>
                    {!isCollapsed ? (
                      <div>
                      <p className="font-robit whitespace-nowrap text-lg font-bold text-[#023e8a]">NUST Bank</p>
                      <p className="whitespace-nowrap text-xs text-slate-500">Precision Banking</p>
                      </div>
                    ) : null}
                  </div>
                  <SidebarTrigger />
                </div>

                <div className="flex h-full flex-col">
                  <Button className="h-12 justify-start gap-2 rounded-2xl bg-[#023e8a] px-4 text-sm font-semibold text-white hover:bg-[#023e8a]/90">
                    <HugeiconsIcon icon={ChatAdd01Icon} size={26} strokeWidth={2.5} />
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
                          className="h-auto w-full justify-start rounded-2xl px-3.5 py-3 text-left text-sm leading-relaxed text-slate-500 hover:bg-[#023e8a]/5 hover:text-slate-700"
                        >
                          {item}
                        </Button>
                      </motion.div>
                    ))}
                  </div> : null}
                </ScrollArea>

                <div className="mt-auto space-y-2 pt-4">
                 
                    <Button variant="ghost" className="h-11 w-full justify-start gap-2.5 rounded-2xl px-3.5 text-slate-600 hover:bg-rose-50 hover:text-rose-500">
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
        className="flex min-h-[92vh] flex-1 flex-col rounded-[34px] border border-white/70 bg-white/65 shadow-[0_18px_70px_-35px_rgba(2,62,138,0.7)] backdrop-blur-xl"
      >
          <motion.header
            className="flex items-center justify-between px-4 py-4 sm:px-7"
            initial={{ y: -16, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.45 }}
          >
            <div className="flex items-center gap-3 lg:hidden">
              <Button variant="outline" size="icon-sm" className="rounded-xl border-[#023e8a]/20 bg-white/90 text-[#023e8a]">
                <HugeiconsIcon icon={ChatIcon} size={16} />
              </Button>
              <p className="font-robit text-base font-bold text-[#023e8a]">NUST Bank</p>
            </div>

            <div className="hidden items-center gap-2 lg:flex">
              <Badge variant="secondary" className="rounded-full px-3 py-1 text-[11px] font-semibold">Assistant</Badge>
              
            </div>

            <div className="flex items-center gap-2 sm:gap-3">
              <Button variant="ghost" size="icon-sm" className="rounded-full text-[#023e8a] hover:bg-[#023e8a]/8">
                <HugeiconsIcon icon={Notification01Icon} size={17} />
              </Button>
              <Avatar className="size-9 border border-[#023e8a]/20">
                <AvatarFallback>MA</AvatarFallback>
              </Avatar>
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
              <Badge className="mb-6 rounded-full bg-[#023e8a]/9 px-4 py-1.5 text-[#023e8a]">Premium Banking Intelligence</Badge>
              <h1 className="font-robit text-center text-4xl leading-tight font-bold text-[#023e8a] sm:text-5xl">
                NUST Bank Assistant
              </h1>
              <p className="mt-5 max-w-2xl text-center text-base leading-relaxed text-slate-600 sm:text-xl">
                Experience the future of precision banking. Securely manage assets, analyze spending, and forecast wealth with your dedicated AI financial partner.
              </p>

              <motion.div
                className="mt-10 w-full"
                initial={{ opacity: 0, scale: 0.97 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true, amount: 0.45 }}
                transition={{ delay: 0.12, duration: 0.45 }}
              >
                <Card className="rounded-[32px] border border-white/70 bg-white/88 p-2 shadow-[0_24px_60px_-30px_rgba(2,62,138,0.75)]">
                  <CardContent className="p-2">
                    <div className="flex items-center gap-3">
                      <Button variant="ghost" size="icon" className="rounded-full bg-[#023e8a]/8 text-[#023e8a] hover:bg-[#023e8a]/12">
                        <HugeiconsIcon icon={Search01Icon} size={18} strokeWidth={1.9} />
                      </Button>
                      <Input
                        className="h-14 border-0 bg-transparent pl-0 text-sm sm:text-base focus-visible:ring-0"
                        placeholder="Ask about your balance, investments, or market trends..."
                      />
                      <Button size="icon-lg" className="rounded-full bg-[#023e8a] text-white shadow-lg shadow-[#023e8a]/30 hover:bg-[#023e8a]/90">
                        <HugeiconsIcon icon={ArrowRight01Icon} size={20} strokeWidth={2} />
                      </Button>
                    </div>
                  </CardContent>
                </Card>
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
                      className="h-10 rounded-full border-[#023e8a]/12 bg-white/70 px-4 text-xs text-[#023e8a] hover:bg-[#023e8a]/7"
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
