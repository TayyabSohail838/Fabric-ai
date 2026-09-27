const fs = require('fs');
const path = require('path');

const url = process.env.SUPABASE_URL;
const anonKey = process.env.SUPABASE_ANON_KEY;

if (url && anonKey) {
  const content = `/* Generated during build from environment variables */
window.FABRIC_SUPABASE = {
  url: ${JSON.stringify(url)},
  anonKey: ${JSON.stringify(anonKey)},
  table: "captures",
  bucket: "fabric-captures",
};
`;
  const target = path.join(__dirname, 'static', 'supabase-config.js');
  fs.writeFileSync(target, content, 'utf8');
  console.log('[build] Generated static/supabase-config.js from environment variables.');
} else {
  console.log('[build] SUPABASE_URL / SUPABASE_ANON_KEY not set in environment.');
}
