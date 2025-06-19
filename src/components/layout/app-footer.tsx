export function AppFooter() {
  return (
    <footer className="py-6 mt-auto border-t">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 text-center text-muted-foreground text-sm">
        © {new Date().getFullYear()} Document Vector Query. All rights reserved.
      </div>
    </footer>
  );
}
