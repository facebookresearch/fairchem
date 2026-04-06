import {
  __commonJS
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/highlight.js/lib/languages/abnf.js
var require_abnf = __commonJS({
  "../../node_modules/highlight.js/lib/languages/abnf.js"(exports, module) {
    function source(re) {
      if (!re)
        return null;
      if (typeof re === "string")
        return re;
      return re.source;
    }
    function concat(...args) {
      const joined = args.map((x) => source(x)).join("");
      return joined;
    }
    function abnf(hljs) {
      const regexes = {
        ruleDeclaration: /^[a-zA-Z][a-zA-Z0-9-]*/,
        unexpectedChars: /[!@#$^&',?+~`|:]/
      };
      const keywords = [
        "ALPHA",
        "BIT",
        "CHAR",
        "CR",
        "CRLF",
        "CTL",
        "DIGIT",
        "DQUOTE",
        "HEXDIG",
        "HTAB",
        "LF",
        "LWSP",
        "OCTET",
        "SP",
        "VCHAR",
        "WSP"
      ];
      const commentMode = hljs.COMMENT(/;/, /$/);
      const terminalBinaryMode = {
        className: "symbol",
        begin: /%b[0-1]+(-[0-1]+|(\.[0-1]+)+){0,1}/
      };
      const terminalDecimalMode = {
        className: "symbol",
        begin: /%d[0-9]+(-[0-9]+|(\.[0-9]+)+){0,1}/
      };
      const terminalHexadecimalMode = {
        className: "symbol",
        begin: /%x[0-9A-F]+(-[0-9A-F]+|(\.[0-9A-F]+)+){0,1}/
      };
      const caseSensitivityIndicatorMode = {
        className: "symbol",
        begin: /%[si]/
      };
      const ruleDeclarationMode = {
        className: "attribute",
        begin: concat(regexes.ruleDeclaration, /(?=\s*=)/)
      };
      return {
        name: "Augmented Backus-Naur Form",
        illegal: regexes.unexpectedChars,
        keywords,
        contains: [
          ruleDeclarationMode,
          commentMode,
          terminalBinaryMode,
          terminalDecimalMode,
          terminalHexadecimalMode,
          caseSensitivityIndicatorMode,
          hljs.QUOTE_STRING_MODE,
          hljs.NUMBER_MODE
        ]
      };
    }
    module.exports = abnf;
  }
});
export default require_abnf();
//# sourceMappingURL=/build/_shared/abnf-C33QBXSX.js.map
