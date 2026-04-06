import {
  __commonJS
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/highlight.js/lib/languages/gherkin.js
var require_gherkin = __commonJS({
  "../../node_modules/highlight.js/lib/languages/gherkin.js"(exports, module) {
    function gherkin(hljs) {
      return {
        name: "Gherkin",
        aliases: ["feature"],
        keywords: "Feature Background Ability Business Need Scenario Scenarios Scenario Outline Scenario Template Examples Given And Then But When",
        contains: [
          {
            className: "symbol",
            begin: "\\*",
            relevance: 0
          },
          {
            className: "meta",
            begin: "@[^@\\s]+"
          },
          {
            begin: "\\|",
            end: "\\|\\w*$",
            contains: [
              {
                className: "string",
                begin: "[^|]+"
              }
            ]
          },
          {
            className: "variable",
            begin: "<",
            end: ">"
          },
          hljs.HASH_COMMENT_MODE,
          {
            className: "string",
            begin: '"""',
            end: '"""'
          },
          hljs.QUOTE_STRING_MODE
        ]
      };
    }
    module.exports = gherkin;
  }
});
export default require_gherkin();
//# sourceMappingURL=/build/_shared/gherkin-PAVW34OH.js.map
