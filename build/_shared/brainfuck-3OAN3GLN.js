import {
  __commonJS
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/highlight.js/lib/languages/brainfuck.js
var require_brainfuck = __commonJS({
  "../../node_modules/highlight.js/lib/languages/brainfuck.js"(exports, module) {
    function brainfuck(hljs) {
      const LITERAL = {
        className: "literal",
        begin: /[+-]/,
        relevance: 0
      };
      return {
        name: "Brainfuck",
        aliases: ["bf"],
        contains: [
          hljs.COMMENT(
            "[^\\[\\]\\.,\\+\\-<> \r\n]",
            "[\\[\\]\\.,\\+\\-<> \r\n]",
            {
              returnEnd: true,
              relevance: 0
            }
          ),
          {
            className: "title",
            begin: "[\\[\\]]",
            relevance: 0
          },
          {
            className: "string",
            begin: "[\\.,]",
            relevance: 0
          },
          {
            // this mode works as the only relevance counter
            begin: /(?:\+\+|--)/,
            contains: [LITERAL]
          },
          LITERAL
        ]
      };
    }
    module.exports = brainfuck;
  }
});
export default require_brainfuck();
//# sourceMappingURL=/build/_shared/brainfuck-3OAN3GLN.js.map
