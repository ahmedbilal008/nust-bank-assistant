import { useState, useRef } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { toast } from "sonner"
import { HugeiconsIcon } from "@hugeicons/react"
import {
  Attachment01Icon,
  MessageQuestionIcon,
  DocumentAttachmentIcon,
  FileEditIcon,
  CloudUploadIcon,
  FactoryIcon,
} from "@hugeicons/core-free-icons"

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { cn } from "@/lib/utils"

const BASE_URL = "https://curblike-theologically-lavelle.ngrok-free.dev"

const HEADERS = {
  "ngrok-skip-browser-warning": "69420",
}

type Tab = "faq" | "text" | "file"

interface Props {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function IngestDialog({ open, onOpenChange }: Props) {
  const [tab, setTab] = useState<Tab>("faq")
  const [isLoading, setIsLoading] = useState(false)

  // FAQ state
  const [faqProduct, setFaqProduct] = useState("")
  const [faqSource, setFaqSource] = useState("")
  const [faqQuestion, setFaqQuestion] = useState("")
  const [faqAnswer, setFaqAnswer] = useState("")

  // Text state
  const [textProduct, setTextProduct] = useState("")
  const [textSource, setTextSource] = useState("")
  const [textContent, setTextContent] = useState("")

  // File state
  const [fileProduct, setFileProduct] = useState("")
  const [fileSource, setFileSource] = useState("")
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const resetAll = () => {
    setFaqProduct(""); setFaqSource(""); setFaqQuestion(""); setFaqAnswer("")
    setTextProduct(""); setTextSource(""); setTextContent("")
    setFileProduct(""); setFileSource(""); setSelectedFile(null)
    if (fileInputRef.current) fileInputRef.current.value = ""
  }

  const handleFAQSubmit = async () => {
    if (!faqQuestion.trim() || !faqAnswer.trim()) {
      toast.error("Please fill in both Question and Answer fields.")
      return
    }
    setIsLoading(true)
    try {
      const body: Record<string, string> = {
        question: faqQuestion.trim(),
        answer: faqAnswer.trim(),
      }
      if (faqProduct.trim()) body.product = faqProduct.trim()
      if (faqSource.trim()) body.source = faqSource.trim()

      const res = await fetch(`${BASE_URL}/ingest`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...HEADERS },
        body: JSON.stringify(body),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data?.detail ?? "Upload failed")
      toast.success(data.message ?? "FAQ ingested successfully!", {
        description: `${data.chunks_added} chunk(s) added · ${data.total_index_vectors} total vectors`,
        position: "bottom-right",
        duration: 5000,
      })
      resetAll()
      onOpenChange(false)
    } catch (err: unknown) {
      toast.error((err instanceof Error ? err.message : "Something went wrong"), { position: "bottom-right" })
    } finally {
      setIsLoading(false)
    }
  }

  const handleTextSubmit = async () => {
    if (!textContent.trim()) {
      toast.error("Please enter some text content.")
      return
    }
    setIsLoading(true)
    try {
      const body: Record<string, string> = { text: textContent.trim() }
      if (textProduct.trim()) body.product = textProduct.trim()
      if (textSource.trim()) body.source = textSource.trim()

      const res = await fetch(`${BASE_URL}/ingest`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...HEADERS },
        body: JSON.stringify(body),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data?.detail ?? "Upload failed")
      toast.success(data.message ?? "Text ingested successfully!", {
        description: `${data.chunks_added} chunk(s) added · ${data.total_index_vectors} total vectors`,
        position: "bottom-right",
        duration: 5000,
      })
      resetAll()
      onOpenChange(false)
    } catch (err: unknown) {
      toast.error((err instanceof Error ? err.message : "Something went wrong"), { position: "bottom-right" })
    } finally {
      setIsLoading(false)
    }
  }

  const handleFileSubmit = async () => {
    if (!selectedFile) {
      toast.error("Please select a .docx or .xlsx file.")
      return
    }
    setIsLoading(true)
    try {
      const formData = new FormData()
      formData.append("file", selectedFile)
      if (fileProduct.trim()) formData.append("product", fileProduct.trim())
      if (fileSource.trim()) formData.append("source", fileSource.trim())

      const res = await fetch(`${BASE_URL}/ingest/file`, {
        method: "POST",
        headers: { ...HEADERS },
        body: formData,
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data?.detail ?? "Upload failed")
      toast.success(data.message ?? "File ingested successfully!", {
        description: `${data.chunks_added} chunk(s) added · ${data.total_index_vectors} total vectors`,
        position: "bottom-right",
        duration: 5000,
      })
      resetAll()
      onOpenChange(false)
    } catch (err: unknown) {
      toast.error((err instanceof Error ? err.message : "Something went wrong"), { position: "bottom-right" })
    } finally {
      setIsLoading(false)
    }
  }

  const handleSubmit = () => {
    if (tab === "faq") handleFAQSubmit()
    else if (tab === "text") handleTextSubmit()
    else handleFileSubmit()
  }

  const tabs: { id: Tab; label: string; icon: typeof MessageQuestionIcon }[] = [
    { id: "faq", label: "FAQ Entry", icon: MessageQuestionIcon },
    { id: "text", label: "Policy Text", icon: FileEditIcon },
    { id: "file", label: "Word / Excel", icon: DocumentAttachmentIcon },
  ]

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className="w-full max-w-lg rounded-2xl border border-primary/15 bg-white/90 p-0 shadow-2xl backdrop-blur-2xl dark:bg-slate-900/90"
        showCloseButton
      >
        {/* Header */}
        <DialogHeader className="px-6 pt-6 pb-0">
          <div className="mb-1 flex items-center gap-2">
            <div className="flex size-8 items-center justify-center rounded-full bg-primary/10">
              <HugeiconsIcon icon={CloudUploadIcon} size={16} strokeWidth={2} className="text-primary" />
            </div>
            <DialogTitle className="text-base font-semibold text-slate-800 dark:text-slate-100">
              Upload Docs / FAQ / Text
            </DialogTitle>
          </div>
          <DialogDescription className="text-xs text-slate-500 dark:text-slate-400">
            Ingest a FAQ entry, freeform policy text, or upload a Word/Excel file to expand the knowledge base.
          </DialogDescription>
        </DialogHeader>

        {/* Tab switcher */}
        <div className="mt-4 flex gap-1 border-b border-primary/10 px-6">
          {tabs.map(({ id, label, icon }) => (
            <button
              key={id}
              onClick={() => setTab(id)}
              className={cn(
                "flex items-center gap-1.5 px-3 py-2 text-xs font-medium transition-colors rounded-t-lg -mb-px border-b-2",
                tab === id
                  ? "border-primary text-primary"
                  : "border-transparent text-slate-400 hover:text-slate-600 dark:hover:text-slate-300"
              )}
            >
              <HugeiconsIcon icon={icon} size={13} strokeWidth={2} />
              {label}
            </button>
          ))}
        </div>

        {/* Tab content */}
        <div className="px-6 pt-4 pb-6">
          <AnimatePresence mode="wait">
            {tab === "faq" && (
              <motion.div
                key="faq"
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -6 }}
                transition={{ duration: 0.18 }}
                className="flex flex-col gap-3"
              >
                <div className="grid grid-cols-2 gap-3">
                  <FieldGroup label="Product (optional)">
                    <Input
                      placeholder="e.g. Home Loans"
                      value={faqProduct}
                      onChange={e => setFaqProduct(e.target.value)}
                    />
                  </FieldGroup>
                  <FieldGroup label="Source (optional)">
                    <Input
                      placeholder="e.g. faq"
                      value={faqSource}
                      onChange={e => setFaqSource(e.target.value)}
                    />
                  </FieldGroup>
                </div>
                <FieldGroup label="Question *">
                  <Input
                    placeholder="What is the maximum tenure for a home loan?"
                    value={faqQuestion}
                    onChange={e => setFaqQuestion(e.target.value)}
                  />
                </FieldGroup>
                <FieldGroup label="Answer *">
                  <Textarea
                    placeholder="NUST Bank offers home loans with a maximum tenure of 20 years..."
                    rows={4}
                    value={faqAnswer}
                    onChange={e => setFaqAnswer(e.target.value)}
                    className="resize-none"
                  />
                </FieldGroup>
              </motion.div>
            )}

            {tab === "text" && (
              <motion.div
                key="text"
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -6 }}
                transition={{ duration: 0.18 }}
                className="flex flex-col gap-3"
              >
                <div className="grid grid-cols-2 gap-3">
                  <FieldGroup label="Product (optional)">
                    <Input
                      placeholder="e.g. Branch Policies"
                      value={textProduct}
                      onChange={e => setTextProduct(e.target.value)}
                    />
                  </FieldGroup>
                  <FieldGroup label="Source (optional)">
                    <Input
                      placeholder="e.g. policy-update-2025"
                      value={textSource}
                      onChange={e => setTextSource(e.target.value)}
                    />
                  </FieldGroup>
                </div>
                <FieldGroup label="Policy / Article Text *">
                  <Textarea
                    placeholder="All NUST Bank branches will operate on Saturdays from 10am to 2pm..."
                    rows={6}
                    value={textContent}
                    onChange={e => setTextContent(e.target.value)}
                    className="resize-none"
                  />
                </FieldGroup>
              </motion.div>
            )}

            {tab === "file" && (
              <motion.div
                key="file"
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -6 }}
                transition={{ duration: 0.18 }}
                className="flex flex-col gap-3"
              >
                <div className="grid grid-cols-2 gap-3">
                  <FieldGroup label="Product (optional)">
                    <Input
                      placeholder="e.g. Home Loan Policy"
                      value={fileProduct}
                      onChange={e => setFileProduct(e.target.value)}
                    />
                  </FieldGroup>
                  <FieldGroup label="Source (optional)">
                    <Input
                      placeholder="e.g. dynamic"
                      value={fileSource}
                      onChange={e => setFileSource(e.target.value)}
                    />
                  </FieldGroup>
                </div>

                {/* Drop zone */}
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className={cn(
                    "group mt-1 flex min-h-[130px] w-full cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed transition-colors",
                    selectedFile
                      ? "border-primary/40 bg-primary/5"
                      : "border-primary/20 bg-primary/3 hover:border-primary/50 hover:bg-primary/8"
                  )}
                >
                  {selectedFile ? (
                    <>
                      <div className="flex size-10 items-center justify-center rounded-full bg-primary/10">
                        <HugeiconsIcon icon={FactoryIcon} size={20} className="text-primary" strokeWidth={1.8} />
                      </div>
                      <p className="max-w-[250px] truncate text-xs font-medium text-primary">
                        {selectedFile.name}
                      </p>
                      <p className="text-[11px] text-slate-400">
                        {(selectedFile.size / 1024).toFixed(1)} KB · Click to change
                      </p>
                    </>
                  ) : (
                    <>
                      <div className="flex size-10 items-center justify-center rounded-full bg-primary/8 transition group-hover:bg-primary/12">
                        <HugeiconsIcon icon={Attachment01Icon} size={20} className="text-primary/60" strokeWidth={1.8} />
                      </div>
                      <p className="text-xs font-medium text-slate-600 dark:text-slate-300">
                        Click to browse file
                      </p>
                      <p className="text-[11px] text-slate-400">Supports .docx and .xlsx</p>
                    </>
                  )}
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".docx,.xlsx"
                  className="hidden"
                  onChange={e => setSelectedFile(e.target.files?.[0] ?? null)}
                />
              </motion.div>
            )}
          </AnimatePresence>

          {/* Submit */}
          <Button
            className="mt-5 w-full rounded-full bg-primary text-primary-foreground shadow-md hover:bg-primary/90 transition cursor-pointer"
            onClick={handleSubmit}
            disabled={isLoading}
          >
            {isLoading ? (
              <span className="flex items-center gap-2">
                <span className="inline-flex gap-1">
                  <span className="size-1.5 animate-bounce rounded-full bg-white/70" style={{ animationDelay: "0ms" }} />
                  <span className="size-1.5 animate-bounce rounded-full bg-white/70" style={{ animationDelay: "150ms" }} />
                  <span className="size-1.5 animate-bounce rounded-full bg-white/70" style={{ animationDelay: "300ms" }} />
                </span>
                Uploading…
              </span>
            ) : (
              <span className="flex items-center gap-2">
                <HugeiconsIcon icon={CloudUploadIcon} size={15} strokeWidth={2.2} />
                Ingest to Knowledge Base
              </span>
            )}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  )
}

/* ── Small helper ── */
function FieldGroup({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-1.5">
      <label className="text-[11px] font-semibold uppercase tracking-wide text-slate-400 dark:text-slate-500">
        {label}
      </label>
      {children}
    </div>
  )
}
