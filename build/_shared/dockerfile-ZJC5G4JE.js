import {
  __commonJS
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/highlight.js/lib/languages/dockerfile.js
var require_dockerfile = __commonJS({
  "../../node_modules/highlight.js/lib/languages/dockerfile.js"(exports, module) {
    function dockerfile(hljs) {
      return {
        name: "Dockerfile",
        aliases: ["docker"],
        case_insensitive: true,
        keywords: "from maintainer expose env arg user onbuild stopsignal",
        contains: [
          hljs.HASH_COMMENT_MODE,
          hljs.APOS_STRING_MODE,
          hljs.QUOTE_STRING_MODE,
          hljs.NUMBER_MODE,
          {
            beginKeywords: "run cmd entrypoint volume add copy workdir label healthcheck shell",
            starts: {
              end: /[^\\]$/,
              subLanguage: "bash"
            }
          }
        ],
        illegal: "</"
      };
    }
    module.exports = dockerfile;
  }
});
export default require_dockerfile();
//# sourceMappingURL=/build/_shared/dockerfile-ZJC5G4JE.js.map
