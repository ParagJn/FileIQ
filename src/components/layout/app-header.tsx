import Link from 'next/link';
import { DocumentSearchIcon } from '@/components/icons/document-search-icon';

export function AppHeader() {
  return (
    <header className="py-4 sm:py-6 border-b bg-card shadow-sm">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 flex items-center gap-3 sm:gap-4">
        <Link href="/" aria-label="Homepage">
          <DocumentSearchIcon className="h-8 w-8 sm:h-10 sm:w-10 text-primary" />
        </Link>
        <Link href="/" className="no-underline hover:no-underline">
          <h1 className="text-xl sm:text-2xl lg:text-3xl font-bold font-headline text-primary">
            FileIQ
          </h1>
        </Link>
      </div>
    </header>
  );
}
