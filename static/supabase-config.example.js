/* Supabase connection for the capture log.
   Copy this file to supabase-config.js (same folder) and fill in both values from
   Supabase > Project Settings > API. supabase-config.js is gitignored.

   Use the anon / publishable key only. Never put the service_role / secret key
   here - this file is served to every visitor's browser.

   Leave the file out, or either value empty, and the app runs exactly as before
   with captures kept only in the page and the CSV export. */
window.FABRIC_SUPABASE = {
  url: "",      // e.g. https://abcdefghijklmnop.supabase.co
  anonKey: "",  // the anon / publishable key
  table: "captures",
};
