import * as React from "react"

import { cn } from "@/lib/utils"

function Avatar({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="avatar"
      className={cn("relative flex size-10 shrink-0 overflow-hidden rounded-full", className)}
      {...props}
    />
  )
}

function AvatarImage({ className, ...props }: React.ComponentProps<"img">) {
  return <img data-slot="avatar-image" className={cn("aspect-square size-full", className)} {...props} />
}

function AvatarFallback({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="avatar-fallback"
      className={cn("bg-[#023e8a]/10 text-[#023e8a] flex size-full items-center justify-center rounded-full text-sm font-semibold", className)}
      {...props}
    />
  )
}

export { Avatar, AvatarImage, AvatarFallback }
