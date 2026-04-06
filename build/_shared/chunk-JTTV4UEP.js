import {
  __commonJS
} from "/build/_shared/chunk-CGOEG7L2.js";

// ../../node_modules/refractor/lang/processing.js
var require_processing = __commonJS({
  "../../node_modules/refractor/lang/processing.js"(exports, module) {
    module.exports = processing;
    processing.displayName = "processing";
    processing.aliases = [];
    function processing(Prism) {
      Prism.languages.processing = Prism.languages.extend("clike", {
        keyword: /\b(?:break|case|catch|class|continue|default|else|extends|final|for|if|implements|import|new|null|private|public|return|static|super|switch|this|try|void|while)\b/,
        // Spaces are allowed between function name and parenthesis
        function: /\b\w+(?=\s*\()/,
        operator: /<[<=]?|>[>=]?|&&?|\|\|?|[%?]|[!=+\-*\/]=?/
      });
      Prism.languages.insertBefore("processing", "number", {
        // Special case: XML is a type
        constant: /\b(?!XML\b)[A-Z][A-Z\d_]+\b/,
        type: {
          pattern: /\b(?:boolean|byte|char|color|double|float|int|[A-Z]\w*)\b/,
          alias: "class-name"
        }
      });
    }
  }
});

export {
  require_processing
};
//# sourceMappingURL=/build/_shared/chunk-JTTV4UEP.js.map
