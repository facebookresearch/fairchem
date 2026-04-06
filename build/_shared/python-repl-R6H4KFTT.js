import {
  __commonJS
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/highlight.js/lib/languages/python-repl.js
var require_python_repl = __commonJS({
  "../../node_modules/highlight.js/lib/languages/python-repl.js"(exports, module) {
    function pythonRepl(hljs) {
      return {
        aliases: ["pycon"],
        contains: [
          {
            className: "meta",
            starts: {
              // a space separates the REPL prefix from the actual code
              // this is purely for cleaner HTML output
              end: / |$/,
              starts: {
                end: "$",
                subLanguage: "python"
              }
            },
            variants: [
              {
                begin: /^>>>(?=[ ]|$)/
              },
              {
                begin: /^\.\.\.(?=[ ]|$)/
              }
            ]
          }
        ]
      };
    }
    module.exports = pythonRepl;
  }
});
export default require_python_repl();
//# sourceMappingURL=/build/_shared/python-repl-R6H4KFTT.js.map
