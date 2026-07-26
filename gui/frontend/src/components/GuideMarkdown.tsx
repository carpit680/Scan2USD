import ReactMarkdown from "react-markdown";
import { Link } from "react-router-dom";

export function GuideMarkdown({ body }: { body: string }) {
  return (
    <div className="prose prose-sm prose-invert max-w-none [&_a]:text-accent [&_code]:text-accent [&_h2]:font-display [&_h2]:text-xl [&_li]:text-ink-300 [&_ol]:text-ink-300 [&_p]:text-ink-300 [&_strong]:text-ink-100">
      <ReactMarkdown
        components={{
          a: ({ href, children }) => {
            if (href && href.startsWith("/")) {
              return (
                <Link to={href} className="text-accent hover:underline">
                  {children}
                </Link>
              );
            }
            return (
              <a
                href={href}
                className="text-accent hover:underline"
                target="_blank"
                rel="noreferrer"
              >
                {children}
              </a>
            );
          },
        }}
      >
        {body}
      </ReactMarkdown>
    </div>
  );
}
