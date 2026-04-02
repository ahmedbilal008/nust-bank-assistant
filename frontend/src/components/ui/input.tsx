import * as React from "react"

import { cn } from "@/lib/utils"

function Input({ className, type = "text", ...props }: React.ComponentProps<"input">) {
  return (
    <input
      type={type}
      data-slot="input"
      className={cn(
        "border-input bg-background/70 placeholder:text-muted-foreground selection:bg-primary selection:text-primary-foreground flex h-11 w-full min-w-0 rounded-full border px-4 py-2 text-sm shadow-xs transition-[color,box-shadow] outline-none file:inline-flex file:h-7 file:border-0 file:bg-transparent file:text-sm file:font-medium disabled:pointer-events-none disabled:cursor-not-allowed disabled:opacity-50 focus-visible:border-[#023e8a]/35 focus-visible:ring-[3px] focus-visible:ring-[#023e8a]/18",
        className
      )}
      {...props}
    />
  )
}

export { Input }
