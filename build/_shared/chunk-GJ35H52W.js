import {
  AbstractMermaidTokenBuilder,
  AbstractMermaidValueConverter,
  ArchitectureGeneratedModule,
  EmptyFileSystem,
  MermaidGeneratedSharedModule,
  __name,
  createDefaultCoreModule,
  createDefaultSharedCoreModule,
  inject,
  lib_exports
} from "/build/_shared/chunk-5TL5RV2T.js";

// ../../node_modules/@mermaid-js/parser/dist/chunks/mermaid-parser.core/chunk-O7ZBX7Z2.mjs
var ArchitectureTokenBuilder = class extends AbstractMermaidTokenBuilder {
  static {
    __name(this, "ArchitectureTokenBuilder");
  }
  constructor() {
    super(["architecture"]);
  }
};
var ArchitectureValueConverter = class extends AbstractMermaidValueConverter {
  static {
    __name(this, "ArchitectureValueConverter");
  }
  runCustomConverter(rule, input, _cstNode) {
    if (rule.name === "ARCH_ICON") {
      return input.replace(/[()]/g, "").trim();
    } else if (rule.name === "ARCH_TEXT_ICON") {
      return input.replace(/["()]/g, "");
    } else if (rule.name === "ARCH_TITLE") {
      return input.replace(/[[\]]/g, "").trim();
    }
    return void 0;
  }
};
var ArchitectureModule = {
  parser: {
    TokenBuilder: /* @__PURE__ */ __name(() => new ArchitectureTokenBuilder(), "TokenBuilder"),
    ValueConverter: /* @__PURE__ */ __name(() => new ArchitectureValueConverter(), "ValueConverter")
  }
};
function createArchitectureServices(context = EmptyFileSystem) {
  const shared = inject(
    createDefaultSharedCoreModule(context),
    MermaidGeneratedSharedModule
  );
  const Architecture = inject(
    createDefaultCoreModule({ shared }),
    ArchitectureGeneratedModule,
    ArchitectureModule
  );
  shared.ServiceRegistry.register(Architecture);
  return { shared, Architecture };
}
__name(createArchitectureServices, "createArchitectureServices");

export {
  ArchitectureModule,
  createArchitectureServices
};
//# sourceMappingURL=/build/_shared/chunk-GJ35H52W.js.map
