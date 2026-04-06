import {
  __commonJS
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/highlight.js/lib/languages/ebnf.js
var require_ebnf = __commonJS({
  "../../node_modules/highlight.js/lib/languages/ebnf.js"(exports, module) {
    function ebnf(hljs) {
      const commentMode = hljs.COMMENT(/\(\*/, /\*\)/);
      const nonTerminalMode = {
        className: "attribute",
        begin: /^[ ]*[a-zA-Z]+([\s_-]+[a-zA-Z]+)*/
      };
      const specialSequenceMode = {
        className: "meta",
        begin: /\?.*\?/
      };
      const ruleBodyMode = {
        begin: /=/,
        end: /[.;]/,
        contains: [
          commentMode,
          specialSequenceMode,
          {
            // terminals
            className: "string",
            variants: [
              hljs.APOS_STRING_MODE,
              hljs.QUOTE_STRING_MODE,
              {
                begin: "`",
                end: "`"
              }
            ]
          }
        ]
      };
      return {
        name: "Extended Backus-Naur Form",
        illegal: /\S/,
        contains: [
          commentMode,
          nonTerminalMode,
          ruleBodyMode
        ]
      };
    }
    module.exports = ebnf;
  }
});
export default require_ebnf();
//# sourceMappingURL=/build/_shared/ebnf-2K7ELIYO.js.map
